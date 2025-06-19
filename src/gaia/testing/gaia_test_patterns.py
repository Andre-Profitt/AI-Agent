import re
from typing import Dict, Any, Callable, List
from dataclasses import dataclass

@dataclass
class GAIATestPattern:
    """GAIA test pattern with validation"""
    name: str
    template: str
    validation: Callable[[Any], bool]
    examples: List[str]
    expected_output_type: str

class GAIATestPatterns:
    """Common GAIA question patterns for testing and validation"""
    
    PATTERNS = {
        "entity_counting": GAIATestPattern(
            name="Entity Counting",
            template="How many {entity} are mentioned in {source}?",
            validation=lambda x: isinstance(x, (int, str)) and str(x).isdigit() and int(x) >= 0,
            examples=[
                "How many albums did The Beatles release?",
                "How many countries are in the European Union?",
                "How many players are on a basketball team?"
            ],
            expected_output_type="integer"
        ),
        
        "date_extraction": GAIATestPattern(
            name="Date Extraction",
            template="When did {event} happen according to {source}?",
            validation=lambda x: bool(re.match(r'\d{4}', str(x))),
            examples=[
                "When was the Declaration of Independence signed?",
                "What year did World War II end?",
                "When did the Berlin Wall fall?"
            ],
            expected_output_type="year"
        ),
        
        "calculation_verification": GAIATestPattern(
            name="Calculation Verification",
            template="Calculate {expression} and verify the result",
            validation=lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').replace('-', '').isdigit()),
            examples=[
                "What is 1729 divided by 7?",
                "Calculate the compound interest on $1000 at 12% for 5 years",
                "What is the square root of 144?"
            ],
            expected_output_type="number"
        ),
        
        "coordinate_extraction": GAIATestPattern(
            name="Coordinate Extraction",
            template="What are the coordinates of {location}?",
            validation=lambda x: bool(re.search(r'\d+\.?\d*[°º]\s*[NS].*\d+\.?\d*[°º]\s*[EW]', str(x))),
            examples=[
                "What are the coordinates of New York City?",
                "Where is the Eiffel Tower located?",
                "What are the GPS coordinates of Tokyo?"
            ],
            expected_output_type="coordinates"
        ),
        
        "country_code_lookup": GAIATestPattern(
            name="Country Code Lookup",
            template="What is the {code_type} code for {country}?",
            validation=lambda x: bool(re.match(r'^[A-Z]{2,3}$', str(x).strip())),
            examples=[
                "What is the ISO 3166-1 alpha-3 code for Egypt?",
                "What is the country code for France?",
                "What is the three-letter code for Japan?"
            ],
            expected_output_type="country_code"
        ),
        
        "chess_move_analysis": GAIATestPattern(
            name="Chess Move Analysis",
            template="What is the best move in this chess position: {position}?",
            validation=lambda x: bool(re.match(r'^[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?$', str(x).strip())),
            examples=[
                "What is the best move for White in the starting position?",
                "What move should Black play after 1.e4 e5 2.Nf3?",
                "What is the correct notation for castling kingside?"
            ],
            expected_output_type="chess_notation"
        ),
        
        "music_discography": GAIATestPattern(
            name="Music Discography",
            template="How many {album_type} did {artist} release between {year1} and {year2}?",
            validation=lambda x: isinstance(x, (int, str)) and str(x).isdigit() and int(x) >= 0,
            examples=[
                "How many studio albums did The Beatles release?",
                "How many albums did Queen release in the 1970s?",
                "How many singles did Michael Jackson release in 1983?"
            ],
            expected_output_type="integer"
        ),
        
        "multi_step_reasoning": GAIATestPattern(
            name="Multi-Step Reasoning",
            template="If {condition1}, and {condition2}, then what is {result}?",
            validation=lambda x: isinstance(x, (int, float, str)) and len(str(x)) > 0,
            examples=[
                "If the Eiffel Tower was built in 1889, and the Statue of Liberty was dedicated 3 years earlier, in what year was the Statue of Liberty dedicated?",
                "If a car travels 60 mph for 2 hours, then 40 mph for 1 hour, what is the average speed?",
                "If Company A has 100 employees and Company B has 50% more, how many employees does Company B have?"
            ],
            expected_output_type="calculated_result"
        )
    }
    
    @classmethod
    def get_pattern_by_question(cls, question: str) -> GAIATestPattern:
        """Identify the most likely pattern for a given question"""
        question_lower = question.lower()
        
        # Score each pattern based on keyword matches
        pattern_scores = {}
        for pattern_name, pattern in cls.PATTERNS.items():
            score = 0
            for example in pattern.examples:
                # Count common words between question and example
                question_words = set(question_lower.split())
                example_words = set(example.lower().split())
                common_words = question_words.intersection(example_words)
                score += len(common_words)
            
            pattern_scores[pattern_name] = score
        
        # Return pattern with highest score
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        return cls.PATTERNS[best_pattern[0]]
    
    @classmethod
    def validate_answer(cls, question: str, answer: Any) -> Dict[str, Any]:
        """Validate answer against expected pattern"""
        pattern = cls.get_pattern_by_question(question)
        
        try:
            is_valid = pattern.validation(answer)
            return {
                "pattern": pattern.name,
                "valid": is_valid,
                "expected_type": pattern.expected_output_type,
                "actual_type": type(answer).__name__,
                "confidence": 0.8 if is_valid else 0.2
            }
        except Exception as e:
            return {
                "pattern": pattern.name,
                "valid": False,
                "error": str(e),
                "expected_type": pattern.expected_output_type,
                "actual_type": type(answer).__name__,
                "confidence": 0.0
            }
    
    @classmethod
    def generate_test_cases(cls) -> List[Dict[str, Any]]:
        """Generate comprehensive test cases for all patterns"""
        test_cases = []
        
        for pattern_name, pattern in cls.PATTERNS.items():
            for example in pattern.examples:
                test_cases.append({
                    "pattern": pattern_name,
                    "question": example,
                    "expected_type": pattern.expected_output_type,
                    "validation": pattern.validation
                })
        
        return test_cases

# Example usage
if __name__ == "__main__":
    patterns = GAIATestPatterns()
    
    # Test pattern identification
    question = "How many studio albums did The Beatles release?"
    pattern = patterns.get_pattern_by_question(question)
    print(f"Question: {question}")
    print(f"Identified Pattern: {pattern.name}")
    print(f"Expected Output Type: {pattern.expected_output_type}")
    
    # Test validation
    test_answer = "13"
    validation_result = patterns.validate_answer(question, test_answer)
    print(f"Validation Result: {validation_result}")
    
    # Generate test cases
    test_cases = patterns.generate_test_cases()
    print(f"Generated {len(test_cases)} test cases") 
# Import and run the new Production-Ready ReAct Agent
from src.app import build_gradio_interface

if __name__ == "__main__":
    print("Launching Production-Ready ReAct Agent...")
    app = build_gradio_interface()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

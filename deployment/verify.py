from deployment.config import EXPORT_PATH

from ai_edge_litert.interpreter import Interpreter


def main():

    print("Loading TensorFlow Lite model...")

    interpreter = Interpreter(
        model_path=str(EXPORT_PATH)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    print("\nModel loaded successfully!")

    print("\n====================================")
    print("Model Information")
    print("====================================")

    print(f"Input Shape : {input_details[0]['shape']}")
    print(f"Input Type  : {input_details[0]['dtype'].__name__}")

    print()

    print(f"Output Shape: {output_details[0]['shape']}")
    print(f"Output Type : {output_details[0]['dtype'].__name__}")

    print()

    print(f"Classes     : {output_details[0]['shape'][1]}")

    print("\n✓ Model verification successful.")

if __name__ == "__main__":
    main()

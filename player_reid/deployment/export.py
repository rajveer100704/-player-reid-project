import sys
import os
# Add project root to path
sys.path.append(os.getcwd())

import torch
import onnx
from player_reid.models.backbone import create_osnet

def export_to_onnx(checkpoint_path, onnx_path, input_size=(1, 3, 256, 128)):
    """
    Export the ReID model to ONNX format for deployment.
    """
    model = create_osnet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dummy_input = torch.randn(input_size)
    
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Export complete.")

if __name__ == "__main__":
    import os
    # Create directory if not exists
    os.makedirs("player_reid/deployment", exist_ok=True)
    export_to_onnx("best_model.pth", "player_reid/deployment/osnet_reid.onnx")

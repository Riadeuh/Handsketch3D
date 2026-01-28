"""
Script d'installation et de configuration de Point-E
"""

import subprocess
import sys
import os


def check_cuda():
    """Verifie si CUDA est disponible"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("CUDA non disponible, utilisation du CPU ")
            return False
    except ImportError:
        print("PyTorch non installe, impossible de verifier CUDA")
        return False


def install_dependencies():
    """Installe les dependances necessaires pour Point-E"""
    print("Installation des dependances Point-E")

    dependencies = [
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "tqdm",
        "clip @ git+https://github.com/openai/CLIP.git",
        "point-e @ git+https://github.com/openai/point-e.git",
    ]

    for dep in dependencies:
        print(f"\n[INSTALL] {dep}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep, "-q"
            ])
            print(f"{dep.split('@')[0].strip()} installé")
        except subprocess.CalledProcessError as e:
            print(f"Echec installation {dep}: {e}")
            return False

    return True


def download_models():
    """Telecharge les modeles Point-E pre-entraines"""
    print("Telechargement des modeles Point-E")

    try:
        import torch
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.models.download import load_checkpoint
        from point_e.models.configs import MODEL_CONFIGS, model_from_config

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        print("\nModele base1B (image-to-3D)...")
        base_name = 'base1B'
        base_model = model_from_config(MODEL_CONFIGS[base_name], device)
        base_model.eval()
        base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

        print("Chargement des poids...")
        base_model.load_state_dict(load_checkpoint(base_name, device))

        print("Modele base1B telecharge")

        print("\nModele upsampler...")
        upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
        upsampler_model.eval()
        upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

        upsampler_model.load_state_dict(load_checkpoint('upsample', device))
        print("Modele upsampler telechargé")

        return True

    except Exception as e:
        print(f"[ERREUR] Echec du telechargement: {e}")
        return False


def verify_installation():
    """Verifie que l'installation est correcte"""

    checks = {
        "torch": False,
        "point_e": False,
        "clip": False,
        "PIL": False,
        "open3d": False,
    }

    try:
        import torch
        checks["torch"] = True
        print(f"[OK] PyTorch {torch.__version__}")
    except ImportError:
        print("[ERREUR] PyTorch non installe")

    try:
        import point_e
        checks["point_e"] = True
        print("[OK] Point-E installe")
    except ImportError:
        print("[ERREUR] Point-E non installe")

    try:
        import clip
        checks["clip"] = True
        print("[OK] CLIP installe")
    except ImportError:
        print("[ERREUR] CLIP non installe")

    try:
        from PIL import Image
        checks["PIL"] = True
        print("[OK] Pillow installe")
    except ImportError:
        print("[ERREUR] Pillow non installe")

    try:
        import open3d
        checks["open3d"] = True
        print(f"[OK] Open3D {open3d.__version__}")
    except ImportError:
        print("[WARNING] Open3D non installe (optionnel pour visualisation)")

    all_required = checks["torch"] and checks["point_e"] and checks["clip"] and checks["PIL"]

    if all_required:
        print("\n[SUCCESS] Installation complete!")
    else:
        print("\n[ERREUR] Installation incomplete")

    return all_required


def main():
    """Point d'entree principal du setup"""

    print("\nVerification de l'installation existante...")
    if verify_installation():
        response = input("\nPoint-E semble deja installe. Reinstaller? (o/n): ")
        if response.lower() != 'o':
            print("Installation annulée.")
            return

    print("\nVerification CUDA...")
    check_cuda()

    print("\nInstallation des dependances...")
    if not install_dependencies():
        print("Echec de l'installation des dependances")
        return

    print("\nTelechargement des modeles...")
    response = input("Telecharger les modeles maintenant? (~300MB) (o/n): ")
    if response.lower() == 'o':
        download_models()
    else:
        print("Vous pouvez telecharger les modeles plus tard avec:")
        print("       python -c \"from Project.PointE.setup_pointe import download_models; download_models()\"")

    verify_installation()


if __name__ == "__main__":
    main()

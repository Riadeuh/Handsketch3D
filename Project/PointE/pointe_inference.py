"""
Wrapper pour l'inference Point-E (Image -> Point Cloud 3D)
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Union
from pathlib import Path

try:
    from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
    from point_e.diffusion.sampler import PointCloudSampler
    from point_e.models.download import load_checkpoint
    from point_e.models.configs import MODEL_CONFIGS, model_from_config
    from point_e.util.plotting import plot_point_cloud
    POINT_E_AVAILABLE = True
except ImportError:
    POINT_E_AVAILABLE = False
    print("!!! Executez setup_pointe.py d'abord. !!!")


class PointEGenerator:
    """Generateur Point-E pour convertir des images/sketches en point clouds 3D"""

    def __init__(
        self,
        device: Optional[str] = None,
        use_upsampler: bool = True,
        verbose: bool = True
    ):
        """
        Initialise le generateur Point-E

        Args:
            device: 'cuda' ou 'cpu' (auto-detecte si None)
            use_upsampler: Si True, utilise l'upsampler pour plus de details
            verbose: Affiche les messages de progression
        """
        if not POINT_E_AVAILABLE:
            raise RuntimeError(
                "Point-E n'est pas installe. "
                "Executez: python Project/PointE/setup_pointe.py"
            )

        self.verbose = verbose
        self.use_upsampler = use_upsampler

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self._log(f"Device: {self.device}")

        self._load_models()

    def _log(self, message: str):
        """Affiche un message si verbose=True"""
        if self.verbose:
            print(f"[PointE] {message}")

    def _load_models(self):
        """Charge les modeles Point-E"""
        self._log("Chargement des modèles...")

        self._log("Chargement du modèle base40M...")
        self.base_name = 'base40M'
        self.base_model = model_from_config(MODEL_CONFIGS[self.base_name], self.device)
        self.base_model.eval()
        self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[self.base_name])
        self.base_model.load_state_dict(load_checkpoint(self.base_name, self.device))

        if self.use_upsampler:
            self._log("Chargement du modèle upsampler...")
            self.upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], self.device)
            self.upsampler_model.eval()
            self.upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
            self.upsampler_model.load_state_dict(load_checkpoint('upsample', self.device))
        else:
            self.upsampler_model = None
            self.upsampler_diffusion = None

        self._log("Création du sampler...")
        self.sampler = PointCloudSampler(
            device=self.device,
            models=[self.base_model, self.upsampler_model] if self.use_upsampler else [self.base_model],
            diffusions=[self.base_diffusion, self.upsampler_diffusion] if self.use_upsampler else [self.base_diffusion],
            num_points=[1024, 4096 - 1024] if self.use_upsampler else [1024],
            aux_channels=['R', 'G', 'B'],
            guidance_scale=[3.0, 0.0] if self.use_upsampler else [3.0],
            model_kwargs_key_filter=('images', '') if self.use_upsampler else ('images',),
            use_karras=[True, True] if self.use_upsampler else [True],
            karras_steps=[64, 64] if self.use_upsampler else [64],
        )
        self._log("Modèles chargés")

    def generate_from_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        num_inference_steps: int = 64,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genere un point cloud 3D a partir d'une image

        Args:
            image: Image d'entree (chemin, PIL Image, ou numpy array)
            num_inference_steps: Nombre d'etapes de diffusion
            seed: Graine aleatoire (optionnel)

        Returns:
            Tuple (points, colors) - arrays numpy (N, 3)
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize((256, 256), Image.LANCZOS)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self._log(f"Generation en cours ({num_inference_steps} etapes)...")

        samples = None
        for x in self.sampler.sample_batch_progressive(
            batch_size=1,
            model_kwargs=dict(images=[image]),
        ):
            samples = x

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()

        if samples.ndim == 1:
            num_points = len(samples) // 6
            pc = samples.reshape(num_points, 6)
        elif samples.ndim == 2:
            batch_size, total_values = samples.shape
            num_points = total_values // 6
            pc = samples.reshape(batch_size, num_points, 6)[0]
        elif samples.ndim == 3:
            if samples.shape[1] == 6:
                pc = samples[0].T
            else:
                pc = samples[0]
        else:
            raise ValueError(f"Forme inattendue: {samples.shape}")

        points = pc[:, :3]
        colors = pc[:, 3:]
        colors = np.clip(colors, 0, 1)

        self._log(f"Point cloud genere: {len(points)} points")

        return points, colors

    def save_point_cloud(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Union[str, Path],
        format: str = 'ply'
    ):
        """
        Sauvegarde le point cloud dans un fichier

        Args:
            points: Coordonnees XYZ (N, 3)
            colors: Couleurs RGB (N, 3) dans [0, 1]
            output_path: Chemin de sortie
            format: 'ply', 'npy', ou 'obj'
        """
        output_path = Path(output_path)

        if format == 'npy':
            np.savez(output_path.with_suffix('.npz'), points=points, colors=colors)
            self._log(f"Point cloud sauvegarde: {output_path.with_suffix('.npz')}")

        elif format == 'ply':
            self._save_ply(points, colors, output_path.with_suffix('.ply'))
            self._log(f"Point cloud sauvegarde: {output_path.with_suffix('.ply')}")

        elif format == 'obj':
            self._save_obj(points, output_path.with_suffix('.obj'))
            self._log(f"Point cloud sauvegarde: {output_path.with_suffix('.obj')}")

        else:
            raise ValueError(f"Format non supporte: {format}")

    def _save_ply(self, points: np.ndarray, colors: np.ndarray, path: Path):
        """Sauvegarde en format PLY"""
        colors_uint8 = (colors * 255).astype(np.uint8)

        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(len(points)):
                f.write(f"{points[i, 0]} {points[i, 1]} {points[i, 2]} ")
                f.write(f"{colors_uint8[i, 0]} {colors_uint8[i, 1]} {colors_uint8[i, 2]}\n")

    def _save_obj(self, points: np.ndarray, path: Path):
        """Sauvegarde en format OBJ"""
        with open(path, 'w') as f:
            for p in points:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    def visualize(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        title: str = "Point Cloud 3D"
    ):
        """
        Visualise le point cloud avec Open3D

        Args:
            points: Coordonnees XYZ (N, 3)
            colors: Couleurs RGB (N, 3) dans [0, 1] (optionnel)
            title: Titre de la fenetre
        """
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

            if colors is not None:
                colors = np.clip(colors, 0, 1).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=title, width=800, height=600)
            vis.add_geometry(pcd)

            opt = vis.get_render_option()
            opt.point_size = 5.0
            opt.background_color = np.array([0.1, 0.1, 0.1])

            vis.run()
            vis.destroy_window()

        except ImportError:
            self._log("Open3D non installe. Utilisez matplotlib comme alternative.")
            self._visualize_matplotlib(points, colors, title)
        except Exception as e:
            self._log(f"Erreur Open3D: {e}. Utilisation de matplotlib...")
            self._visualize_matplotlib(points, colors, title)

    def _visualize_matplotlib(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        title: str = "Point Cloud 3D"
    ):
        """Visualisation alternative avec matplotlib"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        plt.tight_layout()
        plt.show()


def image_to_pointcloud(
    image_path: str,
    output_path: Optional[str] = None,
    visualize: bool = True,
    use_upsampler: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fonction utilitaire pour convertir rapidement une image en point cloud

    Args:
        image_path: Chemin vers l'image d'entree
        output_path: Chemin de sortie (optionnel)
        visualize: Si True, affiche le resultat
        use_upsampler: Si True, utilise l'upsampler pour plus de details

    Returns:
        Tuple (points, colors)
    """
    generator = PointEGenerator(use_upsampler=use_upsampler)
    points, colors = generator.generate_from_image(image_path)

    if output_path:
        generator.save_point_cloud(points, colors, output_path)

    if visualize:
        generator.visualize(points, colors)

    return points, colors


if __name__ == "__main__":
    print("main")

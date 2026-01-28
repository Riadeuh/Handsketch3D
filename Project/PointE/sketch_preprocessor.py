"""
Preprocesseur de sketches pour Point-E
Prepare les dessins a la main pour une meilleure generation 3D
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
from pathlib import Path


class SketchPreprocessor:
    """Preprocesseur pour ameliorer les sketches avant envoi a Point-E"""

    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        invert_colors: bool = True,
        add_background: bool = True,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        line_color: Tuple[int, int, int] = (0, 0, 0),
        enhance_contrast: bool = True,
        smooth_lines: bool = True
    ):
        """
        Initialise le preprocesseur

        Args:
            target_size: Taille de sortie (largeur, hauteur)
            invert_colors: Inverse les couleurs si sketch sur fond noir
            add_background: Ajoute un fond colore
            background_color: Couleur du fond (RGB)
            line_color: Couleur des lignes (RGB)
            enhance_contrast: Ameliore le contraste
            smooth_lines: Lisse les lignes du sketch
        """
        self.target_size = target_size
        self.invert_colors = invert_colors
        self.add_background = add_background
        self.background_color = background_color
        self.line_color = line_color
        self.enhance_contrast = enhance_contrast
        self.smooth_lines = smooth_lines

    def preprocess(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        sketch_on_black: bool = True
    ) -> Image.Image:
        """
        Pretraite un sketch pour Point-E

        Args:
            image: Image d'entree (chemin, array numpy, ou PIL Image)
            sketch_on_black: True si le sketch est dessine sur fond noir

        Returns:
            Image PIL pretraitee
        """
        img = self._load_image(image)

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()

        if sketch_on_black and self.invert_colors:
            gray = cv2.bitwise_not(gray)

        if self.enhance_contrast:
            gray = self._enhance_contrast(gray)

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.smooth_lines:
            binary = self._smooth_lines(binary)

        img_centered = self._center_and_resize(binary)

        if self.add_background:
            img_rgb = self._apply_colors(img_centered)
        else:
            img_rgb = cv2.cvtColor(img_centered, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(img_rgb)

    def _load_image(self, image: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Charge l'image dans un format numpy array"""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()

        return img

    def _enhance_contrast(self, gray: np.ndarray) -> np.ndarray:
        """Ameliore le contraste avec CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _smooth_lines(self, binary: np.ndarray) -> np.ndarray:
        """Lisse les lignes du sketch"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

        return smoothed

    def _center_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Centre le dessin et redimensionne a la taille cible"""
        coords = cv2.findNonZero(cv2.bitwise_not(img))

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.shape[1], x + w + margin)
            y2 = min(img.shape[0], y + h + margin)

            roi = img[y1:y2, x1:x2]

            max_dim = max(roi.shape[0], roi.shape[1])
            square = np.ones((max_dim, max_dim), dtype=np.uint8) * 255

            y_offset = (max_dim - roi.shape[0]) // 2
            x_offset = (max_dim - roi.shape[1]) // 2
            square[y_offset:y_offset + roi.shape[0], x_offset:x_offset + roi.shape[1]] = roi

            resized = cv2.resize(square, self.target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = np.ones(self.target_size, dtype=np.uint8) * 255

        return resized

    def _apply_colors(self, gray: np.ndarray) -> np.ndarray:
        """Applique les couleurs de fond et de ligne"""
        rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
        rgb[:, :] = self.background_color

        mask = gray < 128
        rgb[mask] = self.line_color

        return rgb

    def preprocess_canvas(
        self,
        canvas: np.ndarray,
        draw_color: Tuple[int, int, int] = (0, 255, 0)
    ) -> Image.Image:
        """
        Pretraite un canvas de dessin (depuis main_pipeline.py)

        Args:
            canvas: Canvas numpy array (BGR ou RGB)
            draw_color: Couleur utilisee pour dessiner

        Returns:
            Image PIL pretraitee pour Point-E
        """
        if len(canvas.shape) == 3 and canvas.shape[2] == 3:
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        else:
            canvas_rgb = canvas

        gray = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        inverted = cv2.bitwise_not(mask)

        centered = self._center_and_resize(inverted)

        if self.add_background:
            rgb = self._apply_colors(centered)
        else:
            rgb = cv2.cvtColor(centered, cv2.COLOR_GRAY2RGB)

        return Image.fromarray(rgb)


class QuickDrawPreprocessor(SketchPreprocessor):
    """Preprocesseur specialise pour les donnees QuickDraw"""

    def __init__(self, **kwargs):
        kwargs.setdefault('invert_colors', False)
        kwargs.setdefault('background_color', (255, 255, 255))
        kwargs.setdefault('line_color', (0, 0, 0))
        super().__init__(**kwargs)

    def preprocess_strokes(
        self,
        strokes: list,
        canvas_size: Tuple[int, int] = (256, 256)
    ) -> Image.Image:
        """
        Convertit les strokes QuickDraw en image

        Args:
            strokes: Liste de strokes [[x_coords, y_coords], ...]
            canvas_size: Taille du canvas

        Returns:
            Image PIL
        """
        canvas = np.ones((*canvas_size, 3), dtype=np.uint8) * 255

        for stroke in strokes:
            x_coords = stroke[0]
            y_coords = stroke[1]

            for i in range(len(x_coords) - 1):
                pt1 = (int(x_coords[i]), int(y_coords[i]))
                pt2 = (int(x_coords[i + 1]), int(y_coords[i + 1]))
                cv2.line(canvas, pt1, pt2, (0, 0, 0), 2)

        return self.preprocess(canvas, sketch_on_black=False)


def preprocess_for_pointe(
    image: Union[str, np.ndarray, Image.Image],
    sketch_on_black: bool = True
) -> Image.Image:
    """
    Fonction utilitaire pour pretraiter rapidement une image

    Args:
        image: Image d'entree
        sketch_on_black: True si le sketch est sur fond noir

    Returns:
        Image PIL prete pour Point-E
    """
    preprocessor = SketchPreprocessor()
    return preprocessor.preprocess(image, sketch_on_black=sketch_on_black)


if __name__ == "__main__":
    print("main")

#!/usr/bin/env python3
"""
Script de test Point-E avec dessin a la main
Deux modes: souris ou hand tracking (webcam)
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Project.PointE.sketch_preprocessor import SketchPreprocessor


class MouseDrawer:
    """Fenetre de dessin a la souris"""

    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.drawing = False
        self.last_point = None
        self.color = (0, 255, 0)
        self.thickness = 4

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            if self.last_point is not None:
                cv2.line(self.canvas, self.last_point, (x, y), self.color, self.thickness)
            self.last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.last_point = None

    def run(self):
        """Lance la fenetre de dessin, retourne le canvas ou None"""
        window_name = "Dessinez votre sketch"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        print("  'g' : Generer le modele 3D")
        print("  'c' : Effacer le dessin")
        print("  '+' : Trait epais  |  '-' : Trait fin")
        print("  'q' : Quitter")

        while True:
            display = self.canvas.copy()

            cv2.rectangle(display, (0, 0), (self.width, 35), (30, 30, 30), -1)
            cv2.putText(display, "Dessinez puis appuyez sur 'g' pour generer le 3D",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.circle(display, (self.width - 30, 18), 12, self.color, -1)
            cv2.circle(display, (self.width - 30, 18), 12, (255, 255, 255), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('g'):
                if np.any(self.canvas > 0):
                    cv2.destroyWindow(window_name)
                    return self.canvas.copy()
                else:
                    print(" Canvas vide, dessinez quelque chose d'abord")

            elif key == ord('c'):
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.last_point = None
                print(" Canvas efface")

            elif key == ord('q'):
                cv2.destroyWindow(window_name)
                return None

            elif key == ord('1'):
                self.color = (0, 0, 255)
                print("Couleur: Rouge")
            elif key == ord('2'):
                self.color = (0, 255, 0)
                print("Couleur: Vert")
            elif key == ord('3'):
                self.color = (255, 0, 0)
                print("Couleur: Bleu")
            elif key == ord('+') or key == ord('='):
                self.thickness = min(20, self.thickness + 1)
                print(f"Epaisseur: {self.thickness}")
            elif key == ord('-'):
                self.thickness = max(1, self.thickness - 1)
                print(f"Epaisseur: {self.thickness}")


def run_hand_drawing():
    """Mode dessin avec hand tracking (webcam), retourne le canvas ou None"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from HandTrackingModule import handDetector
    except ImportError:
        print("HandTrackingModule introuvable.")
        print("Assurez-vous que hand_landmarker.task est present.")
        return None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la camera.")
        return None

    detector = handDetector(mode='VIDEO', maxHands=1)
    canvas = None

    print("  'g' : Generer le modele 3D")
    print("  'c' : Effacer le dessin")
    print("  'q' : Quitter")

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)

        if canvas is None:
            canvas = np.zeros_like(img)

        img = detector.findHands(img, draw=True)
        lmList, bbox = detector.findPosition(img, draw=False)

        if len(lmList) != 0:
            index_tip = detector.getIndexFingerTip()
            if detector.checkDrawingGesture():
                if not detector.is_drawing:
                    detector.startDrawing()
                if index_tip:
                    detector.addDrawPoint(index_tip)
                    cv2.circle(img, index_tip, 10, detector.draw_color, cv2.FILLED)
            else:
                if detector.is_drawing:
                    detector.stopDrawing()
            if detector.checkEraseGesture():
                detector.clearDrawing()
                canvas = np.zeros_like(img)

        canvas = detector.drawOnCanvas(canvas)

        img_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

        status = "DESSIN" if detector.is_drawing else "NORMAL"
        status_color = (0, 255, 0) if detector.is_drawing else (0, 0, 255)
        cv2.putText(img_combined, f"Status: {status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cv2.putText(img_combined, "'g'=3D | 'c'=Effacer | 'q'=Quitter",
                    (10, img_combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Hand Drawing -> Point-E", img_combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            if np.any(canvas > 0):
                result = canvas.copy()
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                return result
            else:
                print("Canvas vide, dessinez quelque chose d'abord")

        elif key == ord('c'):
            detector.clearDrawing()
            canvas = np.zeros_like(img)

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            return None

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return None


def generate_3d(canvas):
    """Prend un canvas dessine et genere le point cloud 3D"""
    print(f"\nPretraitement du sketch...")
    preprocessor = SketchPreprocessor()
    preprocessed = preprocessor.preprocess_canvas(canvas)

    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    sketch_path = output_dir / "last_sketch.png"
    preprocessed.save(sketch_path)
    print(f"Sketch sauvegarde: {sketch_path}")

    print("\nChargement de Point-E...")
    try:
        from Project.PointE.pointe_inference import PointEGenerator, POINT_E_AVAILABLE

        if not POINT_E_AVAILABLE:
            print("Point-E n'est pas installe!")
            print("Executez: python Project/PointE/setup_pointe.py")
            return
    except ImportError:
        print("Point-E n'est pas installe!")
        print("Executez: python Project/PointE/setup_pointe.py")
        return

    generator = PointEGenerator(use_upsampler=True, verbose=True)

    print("\nGeneration du point cloud...")
    points, colors = generator.generate_from_image(preprocessed, num_inference_steps=64)

    print(f"\nPoint cloud genere:")
    print(f"  - Nombre de points: {len(points)}")
    print(f"  - X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  - Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  - Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    pc_path = output_dir / "last_pointcloud"
    generator.save_point_cloud(points, colors, pc_path, format='ply')
    print(f"Point cloud sauvegarde: {pc_path}.ply")

    print("\nVisualisation...")
    generator.visualize(points, colors, title="Point-E - Votre sketch")


def run_draw_loop(mode="mouse"):
    """Boucle dessiner -> generer -> redessiner"""

    while True:
        if mode == "hand":
            canvas = run_hand_drawing()
        else:
            drawer = MouseDrawer()
            canvas = drawer.run()

        if canvas is None:
            print("Session terminee.")
            break

        generate_3d(canvas)

        print("\n" + "-" * 40)
        choice = input("Encore un dessin ? (o/n): ").strip().lower()
        if choice != 'o':
            print("Session terminee.")
            break


def main():
    """Point d'entree principal du script de test"""
    parser = argparse.ArgumentParser(
        description="Test Point-E avec dessin a la main",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Exemples:
        python test_pointe.py --draw
        python test_pointe.py --hand
        python test_pointe.py --image mon_sketch.png
        python test_pointe.py --setup
        """
    )

    parser.add_argument('--draw', action='store_true',
                        help='Dessiner a la souris puis generer le 3D')
    parser.add_argument('--hand', action='store_true',
                        help='Dessiner avec le hand tracking (webcam)')
    parser.add_argument('--image', type=str,
                        help='Tester avec une image existante')
    parser.add_argument('--setup', action='store_true',
                        help='Lancer le setup de Point-E')
    parser.add_argument('--output', type=str, default=None,
                        help='Dossier de sortie pour les resultats')

    args = parser.parse_args()

    if args.setup:
        from Project.PointE.setup_pointe import main as setup_main
        setup_main()
        return

    if args.image:
        print(f"\nChargement de l'image: {args.image}")
        canvas = cv2.imread(args.image)
        if canvas is None:
            print(f"Impossible de charger: {args.image}")
            return
        generate_3d(canvas)
        return

    if args.hand:
        run_draw_loop(mode="hand")
        return

    if args.draw:
        run_draw_loop(mode="mouse")
        return

    run_draw_loop(mode="mouse")


if __name__ == "__main__":
    main()

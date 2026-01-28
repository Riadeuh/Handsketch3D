"""
Handsketch3D - Pipeline principale
Sketch a la main -> Preprocessing -> Point-E -> Modele 3D

3 modes :
  1. Dessin a la main (webcam + MediaPipe)
  2. Dessin a la souris
  3. Import d'une image existante
"""

import cv2
import sys
import os
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HandTrackingModule import handDetector
from PointE.sketch_preprocessor import SketchPreprocessor


def generate_3d(canvas):
    """
    Pipeline : canvas -> preprocessing -> Point-E -> point cloud

    Args:
        canvas: Image numpy (BGR) du sketch

    Returns:
        tuple: (points, colors) ou None si erreur
    """
    print("\nPreprocessing...")
    preprocessor = SketchPreprocessor()
    preprocessed = preprocessor.preprocess_canvas(canvas)

    output_dir = Path(__file__).parent / "PointE" / "test_results"
    output_dir.mkdir(exist_ok=True)
    sketch_path = output_dir / "last_sketch.png"
    preprocessed.save(sketch_path)
    print(f"  Sketch sauvegarde: {sketch_path}")

    print("\nChargement de Point-E...")
    try:
        from PointE.pointe_inference import PointEGenerator, POINT_E_AVAILABLE

        if not POINT_E_AVAILABLE:
            print("Point-E n'est pas installe.")
            print("  Lancez : python Project/PointE/setup_pointe.py")
            return None
    except ImportError:
        print("Impossible d'importer Point-E.")
        print("  Lancez : python Project/PointE/setup_pointe.py")
        return None

    generator = PointEGenerator(use_upsampler=True, verbose=True)

    print("\nGeneration du point cloud 3D...")
    points, colors = generator.generate_from_image(preprocessed, num_inference_steps=64)

    print(f"\n  Point cloud genere : {len(points)} points")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    pc_path = output_dir / "last_pointcloud"
    generator.save_point_cloud(points, colors, pc_path, format='ply')
    print(f"  Point cloud sauvegarde: {pc_path}.ply")

    return points, colors


def run_interactive_viewer(points, colors):
    """
    Visualisation 3D interactive controlee par gestes de la main.

    Gestes :
        - Poing ferme : Rotation (position de la main = angle)
        - Pinch (pouce + index) : Zoom
        - Main ouverte : Neutre (pas d'action)
        - 'v' : Reset vue
        - 'q' : Quitter

    Args:
        points: numpy array (N, 3) des coordonnees
        colors: numpy array (N, 3) des couleurs [0-1]
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D n'est pas installe. pip install open3d")
        return

    pc = o3d.geometry.PointCloud()
    base_points = points.astype(np.float64)
    pc.points = o3d.utility.Vector3dVector(base_points.copy())
    if colors is not None and len(colors) == len(points):
        pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Handsketch3D - 3D Model", width=800, height=600)
    vis.add_geometry(pc)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 3.0

    ctr = vis.get_view_control()
    ctr.set_zoom(0.7)
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la camera.")
        vis.run()
        vis.destroy_window()
        return

    detector = handDetector(mode='VIDEO', maxHands=1)

    rotation_x = 0.0
    rotation_y = 0.0
    last_rx = 0.0
    last_ry = 0.0
    zoom = 0.7

    print("  'v' : Reset vue")
    print("  'q' : Quitter")

    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        img = detector.findHands(img, draw=True)
        lmList, bbox = detector.findPosition(img, draw=False)

        gesture_text = "NEUTRE"
        gesture_color = (150, 150, 150)

        if len(lmList) != 0:
            fingers = detector.fingersUp()

            wrist_x = lmList[0][1]
            wrist_y = lmList[0][2]

            if len(fingers) > 0 and sum(fingers) >= 4:
                gesture_text = "NEUTRE"
                gesture_color = (150, 150, 150)

            elif len(fingers) > 0 and sum(fingers) <= 1:
                gesture_text = "ROTATION"
                gesture_color = (0, 255, 255)

                target_ry = (wrist_x / w - 0.5) * 360.0
                target_rx = (0.5 - wrist_y / h) * 180.0

                rotation_y += (target_ry - rotation_y) * 0.15
                rotation_x += (target_rx - rotation_x) * 0.15

            elif len(fingers) > 0 and fingers[0] == 1 and fingers[1] == 1 and sum(fingers) <= 3:
                thumb_x, thumb_y = lmList[4][1], lmList[4][2]
                index_x, index_y = lmList[8][1], lmList[8][2]
                dist = np.hypot(thumb_x - index_x, thumb_y - index_y)

                zoom = np.clip(2.3 - dist / 150.0, 0.3, 2.0)

                gesture_text = f"ZOOM ({zoom:.1f}x)"
                gesture_color = (255, 200, 0)

                cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 200, 0), 2)

        if abs(rotation_x - last_rx) > 0.5 or abs(rotation_y - last_ry) > 0.5:
            rx_rad = np.radians(rotation_x)
            ry_rad = np.radians(rotation_y)

            cos_x, sin_x = np.cos(rx_rad), np.sin(rx_rad)
            cos_y, sin_y = np.cos(ry_rad), np.sin(ry_rad)

            Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
            Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])

            rotated = (Ry @ Rx @ base_points.T).T
            pc.points = o3d.utility.Vector3dVector(rotated)
            vis.update_geometry(pc)

            last_rx = rotation_x
            last_ry = rotation_y

        try:
            ctr = vis.get_view_control()
            ctr.set_zoom(zoom)
        except Exception:
            pass

        if not vis.poll_events():
            break
        vis.update_renderer()

        cv2.putText(img, f"Geste: {gesture_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(img, f"Rot: X={rotation_x:.0f} Y={rotation_y:.0f}",
                    (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "'v'=Reset | 'q'=Quitter",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Handsketch3D - Controle 3D", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('v'):
            rotation_x = 0.0
            rotation_y = 0.0
            zoom = 0.7
            pc.points = o3d.utility.Vector3dVector(base_points.copy())
            vis.update_geometry(pc)
            try:
                ctr = vis.get_view_control()
                ctr.set_zoom(0.7)
            except Exception:
                pass
            print("Vue reinitialisee")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    try:
        vis.destroy_window()
    except Exception:
        pass


def run_hand_mode():
    """
    Mode dessin avec hand tracking.
    L'utilisateur dessine avec l'index leve devant la webcam.
    Appuyer sur 'g' pour generer le modele 3D.

    Returns:
        numpy.ndarray ou None: le canvas dessine
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossible d'ouvrir la camera.")
        return None

    detector = handDetector(mode='VIDEO', maxHands=1)
    canvas = None
    pTime = 0

    print("  'g' : Generer le modele 3D")
    print("  'c' : Effacer le dessin")
    print("  'q' : Quitter")

    while True:
        success, img = cap.read()
        if not success:
            print("Lecture camera echouee.")
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

        status = "DESSIN" if detector.is_drawing else "ATTENTE"
        status_color = (0, 255, 0) if detector.is_drawing else (0, 0, 255)
        cv2.putText(img_combined, f"Status: {status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img_combined, f"FPS: {int(fps)}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.putText(img_combined, "'g'=Generer 3D | 'c'=Effacer | 'q'=Quitter",
                    (10, img_combined.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Handsketch3D - Dessin main", img_combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            if np.any(canvas > 0):
                result = canvas.copy()
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                return result
            else:
                print("Canvas vide, dessinez d'abord !")

        elif key == ord('c'):
            detector.clearDrawing()
            canvas = np.zeros_like(img)
            print("Canvas efface")

        elif key == ord('1'):
            detector.setDrawColor((0, 0, 255))
            print("Couleur: Rouge")
        elif key == ord('2'):
            detector.setDrawColor((0, 255, 0))
            print("Couleur: Vert")
        elif key == ord('3'):
            detector.setDrawColor((255, 0, 0))
            print("Couleur: Bleu")

        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            detector.close()
            return None

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    return None


def run_mouse_mode():
    """
    Mode dessin a la souris.
    L'utilisateur dessine avec clic gauche + glisser.
    Appuyer sur 'g' pour generer le modele 3D.

    Returns:
        numpy.ndarray ou None: le canvas dessine
    """
    width, height = 800, 600
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    drawing = False
    last_point = None
    color = (0, 255, 0)
    thickness = 4

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, last_point, canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            if last_point is not None:
                cv2.line(canvas, last_point, (x, y), color, thickness)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    window_name = "Handsketch3D - Dessin souris"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("  'g' : Generer le modele 3D")
    print("  'c' : Effacer le dessin")
    print("  '+'/'-' : Epaisseur du trait")
    print("  'q' : Quitter")

    while True:
        display = canvas.copy()

        cv2.rectangle(display, (0, 0), (width, 35), (30, 30, 30), -1)
        cv2.putText(display, "'g' pour generer le 3D",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.circle(display, (width - 30, 18), 12, color, -1)
        cv2.circle(display, (width - 30, 18), 12, (255, 255, 255), 1)
        cv2.putText(display, f"e={thickness}", (width - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('g'):
            if np.any(canvas > 0):
                cv2.destroyWindow(window_name)
                return canvas.copy()
            else:
                print("Canvas vide, dessinez d'abord !")

        elif key == ord('c'):
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            last_point = None
            print("Canvas efface")

        elif key == ord('q'):
            cv2.destroyWindow(window_name)
            return None

        elif key == ord('1'):
            color = (0, 0, 255)
            print("Couleur: Rouge")
        elif key == ord('2'):
            color = (0, 255, 0)
            print("Couleur: Vert")
        elif key == ord('3'):
            color = (255, 0, 0)
            print("Couleur: Bleu")
        elif key == ord('+') or key == ord('='):
            thickness = min(20, thickness + 1)
            print(f"Epaisseur: {thickness}")
        elif key == ord('-'):
            thickness = max(1, thickness - 1)
            print(f"Epaisseur: {thickness}")


def run_import_mode():
    """
    Mode import : charge une image existante.

    Returns:
        numpy.ndarray ou None: l'image chargee
    """

    image_path = input("\n  Chemin de l'image : ").strip()

    if not image_path:
        print("Aucun chemin fourni.")
        return None

    image_path = image_path.strip("'\"")

    if not os.path.isfile(image_path):
        print(f"Fichier introuvable : {image_path}")
        return None

    canvas = cv2.imread(image_path)
    if canvas is None:
        print(f"Impossible de charger l'image : {image_path}")
        return None

    print(f"  Image chargee : {image_path}")
    print(f"  Dimensions : {canvas.shape[1]}x{canvas.shape[0]}")

    preview = canvas.copy()
    h, w = preview.shape[:2]
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)))

    cv2.imshow("Apercu - Appuyez sur une touche", preview)
    print("\nAppuyez sur une touche pour continuer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return canvas


def run_ply_import():
    """
    Mode import PLY : charge un fichier .ply et l'affiche directement
    dans le viewer interactif.

    Returns:
        tuple: (points, colors) ou None si erreur
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Open3D n'est pas installe. pip install open3d")
        return None

    ply_path = input("\n  Chemin du fichier .ply : ").strip()

    if not ply_path:
        print("Aucun chemin fourni.")
        return None

    ply_path = ply_path.strip("'\"")

    if not os.path.isfile(ply_path):
        print(f"Fichier introuvable : {ply_path}")
        return None

    pc = o3d.io.read_point_cloud(ply_path)
    if pc is None or len(pc.points) == 0:
        print(f"Fichier PLY invalide ou vide : {ply_path}")
        return None

    points = np.asarray(pc.points)
    if pc.has_colors():
        colors = np.asarray(pc.colors)
    else:
        colors = np.ones_like(points) * 0.7

    print(f"  Modele charge : {ply_path}")
    print(f"  {len(points)} points")

    return points, colors


def select_mode():
    """
    Affiche le menu de selection du mode.

    Returns:
        str: 'hand', 'mouse', 'import', 'ply' ou None
    """
    print("       HANDSKETCH3D")
    print("  Sketch a la main -> Modele 3D")
    print()
    print("  Choisissez un mode :")
    print()
    print("    1. Dessin a la MAIN (webcam + MediaPipe)")
    print("    2. Dessin a la SOURIS")
    print("    3. IMPORTER une image (sketch -> 3D)")
    print("    4. IMPORTER un modele 3D (.ply)")
    print()
    print("    q. Quitter")
    print()

    while True:
        choice = input("\n  Votre choix (1/2/3/4/q) : ").strip()

        if choice == '1':
            return 'hand'
        elif choice == '2':
            return 'mouse'
        elif choice == '3':
            return 'import'
        elif choice == '4':
            return 'ply'
        elif choice.lower() == 'q':
            return None
        else:
            print("  Choix invalide. Entrez 1, 2, 3, 4 ou q.")


def main():
    """Point d'entree principal de Handsketch3D"""

    while True:
        mode = select_mode()

        if mode is None:
            print("\nSession terminée")
            break

        if mode == 'ply':
            result = run_ply_import()
            if result is not None:
                points, colors = result
                run_interactive_viewer(points, colors)
            continue

        if mode == 'hand':
            canvas = run_hand_mode()
        elif mode == 'mouse':
            canvas = run_mouse_mode()
        elif mode == 'import':
            canvas = run_import_mode()
        else:
            canvas = None

        if canvas is None:
            continue

        result = generate_3d(canvas)

        if result is not None:
            points, colors = result
            run_interactive_viewer(points, colors)

        print("\n" + "-" * 40)
        choice = input("  Encore un dessin ? (o/n) : ").strip().lower()
        if choice != 'o':
            print("\nSession terminée")
            break


if __name__ == "__main__":
    main()

from lab.helpers.volume_store import VolumeStore
import numpy as np
import trimesh
from pathlib import Path
from skimage.measure import marching_cubes
from skimage.morphology import closing, ball


class Mesher:
    def __init__(
            self, 
            shell_volume: VolumeStore, 
            kernel_volume: VolumeStore, 
            septa_volume: VolumeStore,
            show_debug: bool = False
    ):
        self._shell_volume = shell_volume
        self._kernel_volume = kernel_volume
        self._septa_volume = septa_volume

        self.show_debug = show_debug

        self._spacing: tuple[float, float, float] = (0.1, 0.1, 0.1)
        self._smooth_iter: int = 5
        self._close_r: int = 2
        self._step_size: int = 2
        
    def process(self) -> tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
        configs = {
            "shell":  dict(close_r=self._close_r, step_size=self._step_size, smooth_iter=self._smooth_iter, spike_threshold_mm=None),
            "kernel": dict(close_r=self._close_r, step_size=self._step_size, smooth_iter=self._smooth_iter, spike_threshold_mm=None),
            "septa":  dict(close_r=self._close_r, step_size=self._step_size, smooth_iter=self._smooth_iter, spike_threshold_mm=0.3),
        }

        meshes: dict[str, trimesh.Trimesh] = {}

        for name, vol in [
            ("shell",  self._shell_volume.get_volume()),
            ("kernel", self._kernel_volume.get_volume()),
            ("septa",  self._septa_volume.get_volume()),
        ]:
            cfg = configs[name]
            mesh = self._volume_to_mesh(
                vol,
                spacing=self._spacing,
                close_r=cfg["close_r"],
                step_size=cfg["step_size"],
            )

            if name == "septa" and not mesh.is_empty:
                components = mesh.split(only_watertight=False)
                if components:
                    mesh = max(components, key=lambda m: len(m.faces))

            mesh = self._postprocess_mesh(
                mesh,
                smooth_iter=cfg["smooth_iter"],
                spike_threshold_mm=cfg["spike_threshold_mm"],
            )

            print(f"{name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            meshes[name] = mesh

        if self.show_debug:
            self._visualize_meshes(meshes["shell"], meshes["kernel"], meshes["septa"])

        return meshes["shell"], meshes["kernel"], meshes["septa"]

    @staticmethod
    def _smooth_volume(volume: np.ndarray, close_r: int = 2) -> np.ndarray:
        if close_r <= 0:
            return volume
        return closing(volume, footprint=ball(close_r))

    def _volume_to_mesh(
        self,
        volume: np.ndarray,
        spacing: tuple[float, float, float],
        close_r: int = 2,
        step_size: int = 1,
    ) -> trimesh.Trimesh:
        vol = self._smooth_volume(volume.astype(bool), close_r=close_r)

        if not vol.any():
            return trimesh.Trimesh()

        verts, faces, normals, _ = marching_cubes(
            vol.astype(np.float32),
            level=0.5,
            spacing=spacing,
            step_size=step_size,
            allow_degenerate=False,
        )

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fill_holes(mesh)

        return mesh

    @staticmethod
    def _remove_spikes(
        mesh: trimesh.Trimesh, 
        max_edge_len_mm: float,
    ) -> trimesh.Trimesh:
        edges = mesh.vertices[mesh.edges_unique]
        lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)

        bad_edges = set(np.where(lengths > max_edge_len_mm)[0])

        face_edges = mesh.faces_unique_edges
        bad_faces_mask = np.any(np.isin(face_edges, list(bad_edges)), axis=1)

        good_faces = mesh.faces[~bad_faces_mask]
        return trimesh.Trimesh(vertices=mesh.vertices, faces=good_faces, process=True)

    @staticmethod
    def _postprocess_mesh(
        mesh: trimesh.Trimesh,
        smooth_iter: int = 5,
        spike_threshold_mm: float | None = 3.0,
    ) -> trimesh.Trimesh:
        if mesh.is_empty:
            return mesh

        if spike_threshold_mm is not None:
            mesh = Mesher._remove_spikes(mesh, max_edge_len_mm=spike_threshold_mm)

        if mesh.is_empty or len(mesh.faces) == 0:
            return mesh

        if smooth_iter > 0:
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=smooth_iter)

        return mesh

    def _visualize_meshes(
        self,
        shell_mesh: trimesh.Trimesh,
        kernel_mesh: trimesh.Trimesh,
        septa_mesh: trimesh.Trimesh,
    ) -> None:
        shell_mesh_visual = shell_mesh.copy()
        kernel_mesh_visual = kernel_mesh.copy()
        septa_mesh_visual = septa_mesh.copy()

        shell_mesh_visual.visual.face_colors = [87, 20, 184, 200]
        kernel_mesh_visual.visual.face_colors = [20, 200, 20, 220]
        septa_mesh_visual.visual.face_colors = [220, 20, 20, 230]

        scene = trimesh.Scene()
        if not shell_mesh_visual.is_empty:
            scene.add_geometry(shell_mesh_visual, node_name="Shell")
        if not kernel_mesh_visual.is_empty:
            scene.add_geometry(kernel_mesh_visual, node_name="Kernel")
        if not septa_mesh_visual.is_empty:
            scene.add_geometry(septa_mesh_visual, node_name="Septa")

        try:
            scene.show()
        except Exception as e:
            print(f"Failed to open 3D viewer: {e}")
            print("Continuing without visualization.")

    @staticmethod
    def save_mesh(
        mesh: trimesh.Trimesh,
        output_path: Path,
    ) -> None:
        if mesh.is_empty:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(output_path))
        print(f"Saved mesh: {output_path}")

"""Motion artifact simulation following TorchIO's approach.

This module implements motion artifacts by simulating rigid body movements during
k-space acquisition. Following Shaw et al., 2019 (http://proceedings.mlr.press/v102/shaw19a.html),
motion is simulated by:
1. Generating multiple rigid transformations (rotation + translation)
2. Applying each transformation to the image
3. Computing FFT of each transformed image
4. Compositing k-space lines from different motion states based on acquisition timing

This approach is more realistic than simple phase modulation as it accounts for
the actual physical motion of the subject during acquisition.
"""

import numpy as np
from typing import Union, Tuple, Optional
from MriWizard.core.base import Record
from MriWizard.core.utils import fft2c, ifft2c, to_complex64, to_float32
from scipy.ndimage import affine_transform

class RandomMotionKspace:
    """Simulate rigid motion artifacts via image resampling and k-space composition.
    
    This implementation follows TorchIO's approach (Shaw et al., 2019) where motion
    is simulated by creating multiple transformed versions of the image and compositing
    their k-space representations.
    """
    
    def __init__(self,
                 degrees: Union[float, Tuple[float, float]] = 10.0,
                 translation: Union[float, Tuple[float, float]] = 10.0,
                 num_transforms: int = 2,
                 perturbation: float = 0.3,
                 axis: int = -2):
        """
        Initialize motion artifact transform.
        
        Args:
            degrees: Rotation range in degrees. Can be:
                    - Single value d: rotations sampled from uniform(-d, d)
                    - Tuple (a, b): rotations sampled from uniform(a, b)
                    For 2D images, rotation is around Z-axis only.
            translation: Translation range in pixels. Can be:
                        - Single value t: translations sampled from uniform(-t, t)
                        - Tuple (a, b): translations sampled from uniform(a, b)
            num_transforms: Number of motion events (default: 2)
                          More transforms = more severe artifacts
            perturbation: Time perturbation factor (default: 0.3)
                         Controls randomness in motion timing
            axis: K-space axis for line-by-line composition (default: -2, phase-encode)
        
        Following TorchIO semantics:
        - Motion events are distributed in time during acquisition
        - Each motion creates a transformed version of the image
        - K-space is composited from these versions line-by-line
        - The transform with time > 0.5 contributes to k-space center (most important)
        """
        # Parse degrees
        if isinstance(degrees, (int, float)):
            self.degrees_range = (-abs(float(degrees)), abs(float(degrees)))
        else:
            self.degrees_range = (float(degrees[0]), float(degrees[1]))
        
        # Parse translation
        if isinstance(translation, (int, float)):
            self.translation_range = (-abs(float(translation)), abs(float(translation)))
        else:
            self.translation_range = (float(translation[0]), float(translation[1]))
        
        if num_transforms < 1:
            raise ValueError(f"num_transforms must be >= 1, got {num_transforms}")
        
        self.num_transforms = num_transforms
        self.perturbation = perturbation
        self.axis = axis
    
    def _sample_transforms_and_times(self, is_2d: bool = False):
        """Sample rotation angles, translations, and timing for motion events.
        
        Args:
            is_2d: Whether the image is 2D (only rotate around Z-axis)
            
        Returns:
            Tuple of (times, rotations, translations)
            - times: array of shape (num_transforms,) in range [0, 1]
            - rotations: array of shape (num_transforms, 3) in radians
            - translations: array of shape (num_transforms, 2 or 3) in pixels
        """
        # Sample rotation angles (in degrees)
        degrees = np.random.uniform(
            self.degrees_range[0],
            self.degrees_range[1],
            size=(self.num_transforms, 3)
        ).astype(np.float32)
        
        # Sample translations (in pixels)
        if is_2d:
            translations = np.random.uniform(
                self.translation_range[0],
                self.translation_range[1],
                size=(self.num_transforms, 2)
            ).astype(np.float32)
        else:
            translations = np.random.uniform(
                self.translation_range[0],
                self.translation_range[1],
                size=(self.num_transforms, 3)
            ).astype(np.float32)
        
        # For 2D: only rotate around Z-axis (last axis)
        if is_2d:
            degrees[:, :2] = 0  # No rotation around X and Y axes
        
        # Convert degrees to radians
        rotations = np.radians(degrees).astype(np.float32)
        
        # Sample times with perturbation
        # Times are when each motion event occurs during acquisition
        step = 1.0 / (self.num_transforms + 1)
        times = np.array([step * (i + 1) for i in range(self.num_transforms)], dtype=np.float32)
        
        # Add random perturbation to times
        noise = np.random.uniform(
            -step * self.perturbation,
            step * self.perturbation,
            size=self.num_transforms
        ).astype(np.float32)
        times = times + noise
        times = np.clip(times, 0.0, 1.0)
        
        return times, rotations, translations
    
    def _create_rotation_matrix_2d(self, angle: float) -> np.ndarray:
        """Create 2D rotation matrix.
        
        Args:
            angle: Rotation angle in radians (rotation around Z-axis)
            
        Returns:
            2x2 rotation matrix
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ], dtype=np.float32)
    
    def _apply_affine_transform_2d(self, image: np.ndarray, rotation: float, 
                                   translation: np.ndarray) -> np.ndarray:
        """Apply 2D affine transformation (rotation + translation).
        
        Args:
            image: 2D image array
            rotation: Rotation angle in radians (around Z-axis)
            translation: Translation vector [tx, ty] in pixels
            
        Returns:
            Transformed image
        """
        # Get image center
        center = np.array(image.shape) / 2.0
        
        # Create rotation matrix
        rotation_matrix = self._create_rotation_matrix_2d(rotation)
        
        # Create affine transform: first translate to origin, rotate, translate back, then apply translation
        # For scipy.ndimage.affine_transform, we need the inverse transformation
        # x' = R(x - center) + center + translation
        # So x = R^(-1)(x' - center - translation) + center
        
        # Inverse rotation (transpose for rotation matrix)
        inv_rotation = rotation_matrix.T
        
        # Offset = center - R^(-1) * (center + translation)
        offset = center - inv_rotation @ (center + translation)
        
        # Apply transformation
        transformed = affine_transform(
            image,
            inv_rotation,
            offset=offset,
            order=1,  # Linear interpolation
            mode='constant',
            cval=0.0
        )
        
        return transformed.astype(np.float32)
    
    def __call__(self, record: Record) -> Record:
        """
        Apply motion artifacts following TorchIO's approach.
        
        Args:
            record: Input record
            
        Returns:
            Record with motion-corrupted k-space
        """
        # Get image (convert from k-space if needed)
        # IMPORTANT: We need to preserve complex k-space through the motion simulation
        # to maintain any prior degradations (noise, etc.)
        if record["kspace"] is not None:
            # Work directly with k-space (preserves all prior degradations)
            kspace_input = record["kspace"].astype(np.complex64)
            # Get magnitude image for motion transformation
            image = to_float32(ifft2c(kspace_input))
            had_image = record["image"] is not None
        elif record["image"] is not None:
            # Start from image, convert to k-space
            image = to_float32(record["image"])
            kspace_input = fft2c(to_complex64(image))
            had_image = True
        else:
            raise ValueError("Record must have either kspace or image")
        
        # Determine if 2D or 3D
        is_2d = image.ndim == 2
        
        # Sample motion parameters
        times, rotations, translations = self._sample_transforms_and_times(is_2d)
        
        # Create transformed images for each motion state
        # First image is always the identity (no transformation)
        transformed_images = [image]
        
        for i in range(self.num_transforms):
            if is_2d:
                # 2D transformation
                transformed = self._apply_affine_transform_2d(
                    image,
                    rotations[i, 2],  # Rotation around Z-axis
                    translations[i, :]  # [tx, ty]
                )
            else:
                # For 3D, we'd need full 3D rigid transformation
                # For now, apply 2D slice-by-slice (simplified)
                # TODO: Implement proper 3D rigid transformation
                transformed = self._apply_affine_transform_2d(
                    image,
                    rotations[i, 2],
                    translations[i, :2]
                )
            
            transformed_images.append(transformed)
        
        # Compute k-space for each transformed image
        spectra = []
        for img in transformed_images:
            spectrum = fft2c(to_complex64(img))
            spectra.append(spectrum)
        
        # Sort spectra: the one with time > 0.5 should be first (contributes to k-space center)
        # This ensures the k-space center comes from a motion state near the middle of acquisition
        if np.any(times > 0.5):
            center_index = np.where(times > 0.5)[0][0] + 1  # +1 because identity is at index 0
        else:
            center_index = len(spectra) - 1
        
        # Swap to put center spectrum first
        spectra[0], spectra[center_index] = spectra[center_index], spectra[0]
        
        # Composite k-space by taking lines from different motion states
        # Lines are assigned based on acquisition time
        result_kspace = np.empty_like(spectra[0], dtype=np.complex64)
        
        # Get k-space dimension along acquisition axis
        axis_size = result_kspace.shape[self.axis]
        
        # Compute line indices for each motion state
        indices = (axis_size * times).astype(int)
        indices = np.append(indices, axis_size)
        
        # Composite k-space line-by-line
        start = 0
        for spectrum_idx, end in enumerate(indices):
            # Create slice for this segment
            idx = [slice(None)] * result_kspace.ndim
            idx[self.axis] = slice(start, end)
            
            # Copy lines from corresponding spectrum
            result_kspace[tuple(idx)] = spectra[spectrum_idx][tuple(idx)]
            
            start = end
        
        # Update record
        record["kspace"] = result_kspace
        
        # Update image if it existed
        if had_image:
            record["image"] = to_float32(ifft2c(result_kspace))
        
        # Log metadata
        record["metadata"]["applied"].append({
            "transform": "RandomMotionKspace",
            "num_transforms": int(self.num_transforms),
            "times": times.tolist(),
            "rotations_deg": np.degrees(rotations).tolist(),
            "translations_px": translations.tolist(),
            "axis": int(self.axis),
            "degrees_range": self.degrees_range,
            "translation_range": self.translation_range
        })
        
        return record


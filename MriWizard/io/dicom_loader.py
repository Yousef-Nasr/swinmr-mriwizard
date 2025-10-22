"""Loader for DICOM files."""

import numpy as np
from pathlib import Path
from MriWizard.core.base import Record
from MriWizard.core.utils import to_float32, fft2c, normalize

class LoadDICOM:
    """Load DICOM files and extract pixel data and metadata."""
    
    def __init__(self, convert_to_kspace: bool = False):
        """
        Initialize DICOM loader.
        
        Args:
            convert_to_kspace: If True, convert image to k-space via FFT
        """
        self.convert_to_kspace = convert_to_kspace
    
    def load(self, path: str) -> Record:
        """
        Load DICOM file.
        
        Args:
            path: Path to DICOM file
            
        Returns:
            Record with image (and optionally kspace) and metadata
        """
        import pydicom
        
        dcm = pydicom.dcmread(str(path))
        # Robust pixel data loading (handle compressed DICOMs gracefully)
        try:
            image = dcm.pixel_array.astype(np.float32)
        except Exception as e:
            ts = getattr(getattr(dcm, 'file_meta', None), 'TransferSyntaxUID', None)
            is_compressed = False
            try:
                is_compressed = bool(ts.is_compressed) if ts is not None else False
            except Exception:
                is_compressed = False
            if is_compressed:
                raise RuntimeError(
                    "Compressed DICOM detected. Install decoders: 'pip install pylibjpeg pylibjpeg-libjpeg gdcm'"
                ) from e
            raise

        # Apply rescale slope/intercept if present
        slope = float(getattr(dcm, 'RescaleSlope', 1.0))
        intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
        if slope != 1.0 or intercept != 0.0:
            image = image * slope + intercept

        # Normalize to [0,1] (percentile-based for stability)
        image = normalize(image, percentile=99.0)
        
        # Extract relevant metadata
        metadata = {
            "source": str(path),
            "modality": str(dcm.get("Modality", "")),
            "applied": [],
            "normalized": True,
            "rescale_slope": slope,
            "rescale_intercept": intercept
        }
        
        # Extract common MR parameters if available
        for tag, key in [
            ("EchoTime", "TE"),
            ("RepetitionTime", "TR"),
            ("FlipAngle", "flip_angle"),
            ("MagneticFieldStrength", "field_strength"),
            ("SliceThickness", "slice_thickness"),
            ("PixelSpacing", "pixel_spacing"),
        ]:
            if hasattr(dcm, tag):
                metadata[key] = getattr(dcm, tag)
        
        # Convert to k-space if requested
        kspace = None
        if self.convert_to_kspace:
            kspace = fft2c(image.astype(np.complex64))
        
        return {
            "kspace": kspace,
            "image": image,
            "mask": None,
            "metadata": metadata
        }
    
    def __call__(self, record: Record) -> Record:
        """Allow use as a transform in pipeline."""
        return record


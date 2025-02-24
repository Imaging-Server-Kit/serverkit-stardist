from typing import List, Literal, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import skimage
import uvicorn
import imaging_server_kit as serverkit
from csbdeep.utils import normalize
from stardist.models import StarDist2D


class Parameters(BaseModel):
    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D).",
        json_schema_extra={"widget_type": "image"},
    )
    stardist_model_name: Literal["2D_versatile_fluo", "2D_versatile_he"] = Field(
        default="2D_versatile_fluo",
        title="Model",
        description="The model used for nuclei segmentation",
        json_schema_extra={"widget_type": "dropdown"},
    )
    prob_thresh: float = Field(
        default=0.5,
        title="Probability threshold",
        description="Predicted object probability threshold",
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.01,
        },
    )
    nms_thresh: float = Field(
        default=0.4,
        title="Overlap threshold",
        description="Overlapping objects are considered the same when their area/surface overlap exceeds this threshold",
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.01,
        },
    )
    scale: float = Field(
        default=1.0,
        title="Scale",
        description="Scale the input image internally by this factor and rescale the output accordingly (<1 to downsample, >1 to upsample)",
        ge=0.0,
        le=10.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.1,
        },
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        if image_array.ndim == 3:
            # Assuming it's RGB, make sure it has 3 channels
            if image_array.shape[2] != 3:
                raise ValueError("Array has the wrong dimensionality.")
        return image_array


class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str = "stardist",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        image: np.ndarray,
        stardist_model_name: str,
        prob_thresh: float,
        nms_thresh: float,
        scale: float,
        **kwargs,
    ) -> List[tuple]:
        """Instance cell nuclei segmentation using StarDist."""
        model = StarDist2D.from_pretrained(stardist_model_name)

        if (image.shape[0] + image.shape[1]) / 2 < 1024:
            segmentation, polys = model.predict_instances(
                normalize(image),
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                scale=scale,
                **kwargs,
            )
        else:
            segmentation, polys = model.predict_instances_big(
                normalize(image),
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                scale=scale,
                block_size=512,
                min_overlap=64,
                axes="YX",
                return_labels=True,
                **kwargs,
            )

        segmentation_params = {"name": f"{stardist_model_name}_mask"}

        return [
            (segmentation, segmentation_params, "instance_mask"),
        ]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Load one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = Server()
app = server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)

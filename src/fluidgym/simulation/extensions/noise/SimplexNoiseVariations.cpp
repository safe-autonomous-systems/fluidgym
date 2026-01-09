/*
	Copyright 2025 Aleksandra Franz, Nils Thürey

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/

#include "simplex_noise.hpp"

using namespace SimplexNoise;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
	py::enum_<NoiseVariation>(m, "NoiseVariation")
		.value("SIMPLEX", NoiseVariation::SIMPLEX)
		.value("WORLEY", NoiseVariation::WORLEY)
		.value("FRACTAL_BROWNIAN_MOTION", NoiseVariation::FRACTAL_BROWNIAN_MOTION)
		.value("RIDGED", NoiseVariation::RIDGED)
		.value("RIDGED_MULTI_FRACTAL", NoiseVariation::RIDGED_MULTI_FRACTAL)
		.value("GRADIENT", NoiseVariation::GRADIENT)
		.value("GRADIENT_FBM", NoiseVariation::GRADIENT_FBM)
		.value("CURL", NoiseVariation::CURL)
		.value("CURL_FBM", NoiseVariation::CURL_FBM)
		.export_values();
	
	m.def("GenerateSimplexNoiseVariation", &GenerateSimplexNoiseVariation, "Create a tensor with simplex noise.",
		py::arg("output_shape"), py::arg("GPUdevice"), py::arg("scale"), py::arg("offset"), py::arg("variation"),
		py::arg("ridgeOffset")=1.0f, py::arg("octaves")=4, py::arg("lacunarity")=2.0f, py::arg("gain")=0.5f);
}
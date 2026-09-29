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

#pragma once

#ifndef _INCLUDE_OPTIONAL_TYPE
#define _INCLUDE_OPTIONAL_TYPE

#include <torch/extension.h>

template <typename T>
using optional = c10::optional<T>;
const auto nullopt = c10::nullopt;

#endif //_INCLUDE_OPTIONAL_TYPE

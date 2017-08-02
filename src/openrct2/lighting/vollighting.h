#pragma once

#include "../common.h"

typedef struct lighting_chunk {
	float data[16][16][16];
	bool to_rebuild;
	
} lighting_chunk;

typedef struct lighting_static_light {
	float x;
	float y;
	float z;
} lighting_static_light;

void lighting_reset();
void lighting_compute_skylight(sint32 x, sint32 y);
void lighting_apply_light(sint32 x, sint32 y);
void lighting_apply_light_3d(sint32 x, sint32 y, sint32 z);
float* lighting_get_data();
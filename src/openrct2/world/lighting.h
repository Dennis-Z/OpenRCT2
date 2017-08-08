#pragma once

#include "../common.h"
#include "../world/map.h"

#define lightmap_size_x 512
#define lightmap_size_y 512
#define lightmap_size_z 128
#define lightmap_chunk_size 16
#define lightmap_chunks_x (lightmap_size_x / lightmap_chunk_size)
#define lightmap_chunks_y (lightmap_size_y / lightmap_chunk_size)
#define lightmap_chunks_z (lightmap_size_z / lightmap_chunk_size)
#define lighting_max_chunk_lights 64

#pragma pack(push, 1)
typedef struct lighting_value {
	uint8 r;
	uint8 g;
	uint8 b;
} lighting_value;
#pragma pack(pop)

typedef struct lighting_light {
	rct_xyz32 pos;
	lighting_value color;
	sint32 map_x;
	sint32 map_y;
} lighting_light;

typedef struct lighting_chunk {
	lighting_value data[lightmap_chunk_size][lightmap_chunk_size][lightmap_chunk_size];
	lighting_light static_lights[lighting_max_chunk_lights];
	size_t static_lights_count;
	bool invalid;
	uint8 x, y, z;
} lighting_chunk;

void lighting_init();
void lighting_reset();
void lighting_invalidate_at(sint32 x, sint32 y);
void lighting_invalidate_around(sint32 x, sint32 y);
void lighting_invalidate_all();
lighting_chunk* lighting_update();
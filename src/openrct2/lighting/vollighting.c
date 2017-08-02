#include "vollighting.h"

float gMapLighting[64][512][512];

lighting_chunk gLightingChunks[64/16][512/16][512/16];

void lighting_reset() {
	/*for (int z = 0; z < 50; z++) {
	for (int y = 0; y < 256; y++) {
	for (int x = 0; x < 256; x++) {
	gMapLighting[z][y][x] = 0;
	}
	}
	}*/
	memset(gMapLighting, 0, sizeof gMapLighting);
}

void lighting_compute_skylight(sint32 x, sint32 y) {
	/*rct_map_element * mapElement = map_get_first_element_at(x, y);
	if (mapElement) {
	float lght = 0.4f;
	for (int z = 0; z < mapElement->base_height; z++) {
	gMapLighting[z][y][x] = lght;
	}
	for (int z = mapElement->base_height; z < 50; z++) {
	gMapLighting[z][y][x] = lght;
	if (lght < 1) {
	lght += 0.05f;
	}
	}
	}*/
	for (int z = 0; z < 50; z++) {
		gMapLighting[z][y][x] = 0.2f;
	}
}

void lighting_apply_light(sint32 x, sint32 y) {
	sint32 z = 14;
	sint32 range = 5;
	for (int pz = min(0, z - range * 4); pz <= z + range * 4; pz++) {
		for (int py = y - range; py <= y + range; py++) {
			for (int px = x - range; px <= x + range; px++) {
				float distpot = sqrt((px - x)*(px - x) + (py - y)*(py - y) + (pz - z)*0.25*(pz - z)*0.25);
				float intensity = 1.0f - distpot * 0.2f;
				if (intensity < 0) intensity = 0;
				gMapLighting[pz][py][px] += intensity * 0.6f;
			}
		}
	}
}

void lighting_apply_light_3d(sint32 x, sint32 y, sint32 z) {
	sint32 range = 5;
	log_info("create_light at %d %d %d", x, y, z);
	sint32 lm_x = (x + 8) / 16;
	sint32 lm_y = (y + 8) / 16;
	sint32 lm_z = (z + 1) / 2;
	for (int pz = max(0, lm_z - range * 2); pz <= min(63, lm_z + range * 2); pz++) {
		for (int py = max(0, lm_y - range); py <= min(511, lm_y + range); py++) {
			for (int px = max(0, lm_x - range); px <= min(511, lm_x + range); px++) {
				sint32 w_x = px * 16;
				sint32 w_y = py * 16;
				sint32 w_z = pz * 2;
				float distpot = sqrt((w_x - x)*(w_x - x) + (w_y - y)*(w_y - y) + (w_z - z)*(w_z - z) * 4 * 4);
				float intensity = 1.0f - distpot / (range * 16.0f);
				if (intensity < 0) intensity = 0;
				gMapLighting[pz][py][px] += intensity * 0.1f;
			}
		}
	}
}

float* lighting_get_data() {
	return &gMapLighting[0][0][0];
}
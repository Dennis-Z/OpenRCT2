#include "vollighting.h"
#include "../world/map.h"

float gMapLighting[64][512][512];

float lightingAffectorsX[64][512][512];
float lightingAffectorsY[64][512][512];
float lightingAffectorsZ[64][512][512];

lighting_chunk gLightingChunks[64/16][512/16][512/16];


bool didInit2 = false;

float FASTCALL lighting_raycast(rct_xyz16 a, rct_xyz16 b) {
	float x = a.x;
	float y = a.y;
	float z = a.z;
	float dx = (b.x) - x;
	float dy = (b.y) - y;
	float dz = (b.z) - z;
	float result = 1.0;
	//float dmag = sqrt(dx*dx + dy*dy + dz*dz);
	//log_info("raycast");

	for (int px = min(b.x, a.x); px < max(a.x, b.x); px++) {
		float multiplier = (px - a.x + 0.5) / dx;
		float py = dy * multiplier + y;
		float pz = dz * multiplier + z;
		int affectorY = py;
		int affectorZ = pz;
		float progy = fmod(py, 1.0);
		float progz = fmod(pz, 1.0);
		float v00 = lightingAffectorsX[affectorZ][affectorY][px];
		float v01 = lightingAffectorsX[affectorZ][affectorY-1][px];
		float v10 = lightingAffectorsX[affectorZ-1][affectorY][px];
		float v11 = lightingAffectorsX[affectorZ-1][affectorY-1][px];
		float v0 = v00 * progy + v01 * (1.0 - progy);
		float v1 = v10 * progy + v11 * (1.0 - progy);
		result *= v0 * progz + v1 * (1.0 - progz);
	}

	for (int py = min(b.y, a.y); py < max(a.y, b.y); py++) {
		float multiplier = (py - a.y + 0.5) / dy;
		float px = dx * multiplier + x;
		float pz = dz * multiplier + z;
		int affectorX = px;
		int affectorZ = pz;
		float progx = fmod(px, 1.0);
		float progz = fmod(pz, 1.0);
		float v00 = lightingAffectorsY[affectorZ][py][affectorX];
		float v01 = lightingAffectorsY[affectorZ][py][affectorX - 1];
		float v10 = lightingAffectorsY[affectorZ - 1][py][affectorX];
		float v11 = lightingAffectorsY[affectorZ - 1][py][affectorX - 1];
		float v0 = v00 * progx + v01 * (1.0 - progx);
		float v1 = v10 * progx + v11 * (1.0 - progx);
		result *= v0 * progz + v1 * (1.0 - progz);
	}

	//if (b.x >= 60) {
	//	return 0.0;
	//}
	return result;
}

void lighting_init() {
	// reset affectors to 1^3
	for (int z = 0; z < 64; z++) {
		for (int y = 0; y < 512; y++) {
			for (int x = 0; x < 512; x++) {
				lightingAffectorsX[z][y][x] = 1.0;
				lightingAffectorsY[z][y][x] = 1.0;
				lightingAffectorsZ[z][y][x] = 1.0;
			}
		}
		if (z < 15) {
			lightingAffectorsX[z][61][61] = 0.0;
			lightingAffectorsX[z][62][61] = 0.0;
			lightingAffectorsX[z][63][61] = 0.0;
			lightingAffectorsX[z][64][61] = 0.0;
			lightingAffectorsX[z][65][61] = 0.0;
			lightingAffectorsX[z][66][61] = 0.0;
			lightingAffectorsY[z][61][61] = 0.0;
			lightingAffectorsY[z][61][62] = 0.0;
			lightingAffectorsY[z][61][63] = 0.0;
			lightingAffectorsY[z][61][64] = 0.0;

			lightingAffectorsX[z][61 + 8][59] = 0.0;
			lightingAffectorsX[z][62 + 8][59] = 0.0;
			lightingAffectorsX[z][63 + 8][59] = 0.0;
			lightingAffectorsX[z][64 + 8][59] = 0.0;
			lightingAffectorsX[z][65 + 8][59] = 0.0;
			lightingAffectorsX[z][66 + 8][59] = 0.0;
		}
	}
	didInit2 = false;
}

void lighting_reset() {
	if (didInit2) return;
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
	if (didInit2) return;
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
				if (intensity > 0) {
					gMapLighting[pz][py][px] += intensity * 10.6f;
				}
			}
		}
	}
}

void lighting_apply_light_3d(sint32 x, sint32 y, sint32 z) {
	if (didInit2) return;
	sint32 range = 11;
	log_info("create_light at %d %d %d", x, y, z);
	sint32 lm_x = (x + 8) / 16;
	sint32 lm_y = (y + 8) / 16;
	sint32 lm_z = (z + 1) / 2;
	rct_xyz16 source = { .x = lm_x,.y = lm_y,.z = lm_z };
	for (int pz = max(0, lm_z - range * 2); pz <= min(63, lm_z + range * 2); pz++) {
		for (int py = max(0, lm_y - range); py <= min(511, lm_y + range); py++) {
			for (int px = max(0, lm_x - range); px <= min(511, lm_x + range); px++) {
				sint32 w_x = px * 16;
				sint32 w_y = py * 16;
				sint32 w_z = pz * 2;
				float distpot = sqrt((w_x - x)*(w_x - x) + (w_y - y)*(w_y - y) + (w_z - z)*(w_z - z) * 4 * 4);
				float intensity = 1.0f - distpot / (range * 12.0f);
				if (intensity > 0) {
					rct_xyz16 target = { .x = px,.y = py,.z = pz };
					float multiplier = lighting_raycast(source, target);
					gMapLighting[pz][py][px] += multiplier * intensity * 0.1f;
				}
			}
		}
	}
}

float* lighting_get_data() {
	if (didInit2) return 0;
	didInit2 = true;
	return &gMapLighting[0][0][0];
}
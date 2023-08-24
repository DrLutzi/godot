#include "texsyn_tests.h"
#include "../procedural_sampling.h"

bool texsyn_tests()
{
	bool b = true;
	TexSyn::ImageVector<double> imageVector;
	imageVector.init(512, 512, 5, true);
	DEV_ASSERT(b &= imageVector.get_pixel(511, 511, 4) == 0.0);
	imageVector.set_pixel(0, 0, 0, 5.0);
	DEV_ASSERT(b &= imageVector.get_pixel(0, 0, 0) == 5.0);

	TexSyn::ImageVector<double> imageVector2;
	imageVector2.init(512, 512, 5, true);
	imageVector2.set_pixel(0, 0, 0, 2.0);

	imageVector *= 3.0;
	DEV_ASSERT(b &= imageVector.get_pixel(0, 0, 0) == 15.0);
	imageVector /= 3.0;
	DEV_ASSERT(b &= imageVector.get_pixel(0, 0, 0) == 5.0);
	imageVector += 3.0;
	DEV_ASSERT(b &= imageVector.get_pixel(0, 0, 0) == 8.0);
	imageVector -= 3.0;
	DEV_ASSERT(b &= imageVector.get_pixel(0, 0, 0) == 5.0);

	imageVector.set_pixel(511, 0, 2, 8.0);

	imageVector.operator=(imageVector*2.0);
	DEV_ASSERT(b &= imageVector.get_pixel(511, 0, 2) == 16.0);

	imageVector.operator=(imageVector/2.0);
	DEV_ASSERT(b &= imageVector.get_pixel(511, 0, 2) == 8.0);

	imageVector.operator=(imageVector+2.0);
	DEV_ASSERT(b &= imageVector.get_pixel(511, 0, 2) == 10.0);

	imageVector.operator=(imageVector-2.0);
	DEV_ASSERT(b &= imageVector.get_pixel(511, 0, 2) == 8.0);

	DEV_ASSERT(b &= imageVector.get_pixel(0, 0, 0) == 5.0);

	imageVector.set_pixel(0, 0, 1, 4.0);
	imageVector2.set_pixel(0, 0, 1, 4.0);
	DEV_ASSERT(b &= (imageVector *= imageVector2).get_pixel(0, 0, 1) == 16.0);
	DEV_ASSERT(b &= (imageVector += imageVector2).get_pixel(0, 0, 1) == 20.0);
	DEV_ASSERT(b &= (imageVector /= imageVector2).get_pixel(0, 0, 1) == 5.0);
	DEV_ASSERT(b &= (imageVector -= imageVector2).get_pixel(0, 0, 1) == 1.0);

	DEV_ASSERT(b &= (imageVector * imageVector2).get_pixel(0, 0, 1) == 4.0);
	DEV_ASSERT(b &= (imageVector + imageVector2).get_pixel(0, 0, 1) == 5.0);
	DEV_ASSERT(b &= (imageVector - imageVector2).get_pixel(0, 0, 1) == -3.0);
	DEV_ASSERT(b &= (imageVector / imageVector2).get_pixel(0, 0, 1) == 0.25);

	if (b)
		print_line("ImageVector tests : OK.");
		
	return b;
}

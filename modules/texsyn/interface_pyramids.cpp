#include "interface_pyramids.h"

void TexSyn::GaussianPyr::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("init", "image", "depth"), &GaussianPyr::init);
	ClassDB::bind_method(D_METHOD("init_from_pyramid", "pyramid"), &GaussianPyr::init_from_pyramid);
	ClassDB::bind_method(D_METHOD("get_layer", "depth"), &GaussianPyr::get_layer);
}

void TexSyn::LaplacianPyr::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("init", "image", "depth"), &LaplacianPyr::init);
	ClassDB::bind_method(D_METHOD("init_from_pyramid", "pyramid"), &LaplacianPyr::init_from_pyramid);
	ClassDB::bind_method(D_METHOD("get_layer", "depth"), &LaplacianPyr::get_layer);
}

void TexSyn::RieszPyr::_bind_methods()
{
	BIND_ENUM_CONSTANT(CARTESIAN);
	BIND_ENUM_CONSTANT(POLAR);

	ClassDB::bind_method(D_METHOD("init", "image", "depth"), &RieszPyr::init);
	ClassDB::bind_method(D_METHOD("get_layer", "depth", "type"), &RieszPyr::get_layer);
	ClassDB::bind_method(D_METHOD("pack_in_texture", "type"), &RieszPyr::pack_in_texture);
	ClassDB::bind_method(D_METHOD("phases_congruency", "alpha", "beta", "type"), &RieszPyr::phase_congruency);
	ClassDB::bind_method(D_METHOD("reconstruct"), &RieszPyr::reconstruct);
	ClassDB::bind_method(D_METHOD("test", "image", "subImage"), &RieszPyr::test);
}

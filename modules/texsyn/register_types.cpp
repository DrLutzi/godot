#include "image_pyramid.h"
#include "register_types.h"
#include "core/object/class_db.h"
#include "texture_synthesizer.h"

void initialize_texsyn_module(ModuleInitializationLevel p_level)
{
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
	{
		return;
	}
	ClassDB::register_class<TexSyn::GaussianPyr>();
	ClassDB::register_class<TexSyn::LaplacianPyr>();
	ClassDB::register_class<TexSyn::RieszPyr>();

	ClassDB::register_class<TextureSynthesizer>();

#ifdef TEXSYN_TESTS
	DEV_ASSERT(texsyn_tests());
#endif
}

void uninitialize_texsyn_module(ModuleInitializationLevel p_level)
{
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
	{
		return;
	}
}

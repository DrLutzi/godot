#ifndef TEXSYN_H
#define TEXSYN_H

#include "image_scalar.h"
#include "image_vector.h"
#include "texsyn_pca.h"
#include "texsyn_statistics.h"
#include "texsyn_procedural_sampling.h"

constexpr std::uint32_t texsyn_log2(std::uint32_t n) noexcept
{
	return (n > 1) ? 1 + texsyn_log2(n >> 1) : 0;
}

class ImageScalard : public RefCounted
{
	GDCLASS(ImageScalard, RefCounted);

public:

	ImageScalard()
		: RefCounted()
	{}

protected:
	static void _bind_methods();
};


class ImageVectord : public RefCounted
{
	GDCLASS(ImageVectord, RefCounted);

public:
	ImageVectord()
		: RefCounted()
	{}

protected:
	static void _bind_methods();
};

class StatisticsScalard : public RefCounted
{
	GDCLASS(StatisticsScalard, RefCounted);

public:
	StatisticsScalard();
	~StatisticsScalard();

	void init(Ref<ImageScalard> imRef);
	Ref<ImageScalard> get_FourierModulus();

protected:
	static void _bind_methods();

private:

	using StatisticsType = TexSyn::StatisticsScalar<double>;
	StatisticsType *m_statistics;
};

class ProceduralSampling : public RefCounted
{
	GDCLASS(ProceduralSampling, RefCounted);

public:

	enum TextureTypeFlag
	{
		ALBEDO=1,
		NORMAL=2,
		HEIGHT=4,
		ROUGHNESS=8,
		METALLIC=16,
		AMBIENT_OCCLUSION=32,
		SPECULAR=64,
		ALPHA=128,
		RIM=256
	};

	ProceduralSampling();

	void set_albedo(Ref<Image> image);
	void set_normal(Ref<Image> image);
	void set_height(Ref<Image> image);
	void set_roughness(Ref<Image> image);
	void set_metallic(Ref<Image> image);
	void set_ao(Ref<Image> image);

	void set_component(TextureTypeFlag type, Ref<Image> image);

	void spatiallyVaryingMeanToAlbedo(Ref<Image> image);
	void spatiallyVaryingMeanToNormal(Ref<Image> image);
	void spatiallyVaryingMeanToHeight(Ref<Image> image);
	void spatiallyVaryingMeanToRoughness(Ref<Image> image);
	void spatiallyVaryingMeanToMetallic(Ref<Image> image);
	void spatiallyVaryingMeanToAO(Ref<Image> image);

	void spatiallyVaryingMeanToComponent(TextureTypeFlag type, Ref<Image> image);

	void set_cyclostationaryPeriods(Vector2 t0, Vector2 t1);
	void set_importancePDF(Ref<Image> image);
	void set_meanAccuracy(unsigned int accuracy);
	void set_meanSize(unsigned int meanSize);

	void computeAutocovarianceSampler();
	void samplerPdfToImage(Ref<Image> image);

	void samplerRealizationToImage(Ref<Image> image, unsigned int size);

	void centerExemplar(Ref<Image> exemplar, Ref<Image> mean);

protected:
	static void _bind_methods();

private:
	void computeImageVector();

	int m_textureTypeFlag;
	LocalVector<Ref<Image>> m_imageRefs;
	TexSyn::ImageVector<float> m_exemplar;
	TexSyn::ImageVector<float> m_weightedMean;
	TexSyn::ProceduralSampling<float> m_proceduralSampling;
	unsigned int m_meanAccuracy;
	unsigned int m_meanSize;
};

#define TEXSYN_TESTS
#ifdef TEXSYN_TESTS

bool texsyn_tests();

VARIANT_ENUM_CAST(ProceduralSampling::TextureTypeFlag);

#endif //ifdef TEXSYN_TESTS

#endif // TEXSYN_H

#ifndef TEXSYN_H
#define TEXSYN_H

#include "image_scalar.h"
#include "image_vector.h"
#include "pca.h"
#include "statistics.h"
#include "procedural_sampling.h"
#include "scene/resources/image_texture.h"
#include "gaussian_transfer.h"

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

class TextureSynthesizer : public RefCounted
{
	GDCLASS(TextureSynthesizer, RefCounted);

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

	//Base
	
	using ImageVectorType = TexSyn::ImageVector<float>;
	using ImageScalarType = TexSyn::ImageScalar<float>;

	TextureSynthesizer();

	void set_component(TextureTypeFlag type, Ref<Image> image);
	void outputToComponent(TextureTypeFlag type, Ref<Image> image);

protected:
	static void _bind_methods();

	void computeImageVector();
	unsigned int getStartIndexFromComponent(TextureTypeFlag flag);

	int m_textureTypeFlag;
	LocalVector<Ref<Image>> m_imageRefs;
	ImageVectorType m_exemplar;
	ImageVectorType m_outputImageVector;
};

class SamplerTextureSynthesizer : public TextureSynthesizer
{
	GDCLASS(SamplerTextureSynthesizer, TextureSynthesizer);

public:

	SamplerTextureSynthesizer();

	void set_cyclostationaryPeriods(Vector2 t0, Vector2 t1);
	void set_importancePDF(Ref<Image> image);
	void set_meanAccuracy(unsigned int accuracy);
	void set_meanSize(unsigned int meanSize);

	void computeAutocovarianceSampler();
	void samplerPdfToImage(Ref<Image> image);
	void samplerRealizationToImage(Ref<Image> image, unsigned int size);
	void centerExemplar(Ref<Image> exemplar, Ref<Image> mean);
	
	void computeWeightedMean();
	
protected:
	static void _bind_methods();
	
private:
	TexSyn::ProceduralSampling<float> m_proceduralSampling;
	unsigned int m_meanAccuracy;
	unsigned int m_meanSize;
};

class LocallyStationaryTextureSynthesizer : public TextureSynthesizer
{
	GDCLASS(LocallyStationaryTextureSynthesizer, TextureSynthesizer);
	
public:

	using ImageRegionType = TexSyn::ImageScalar<int>;
	using PCAType = TexSyn::PCA<float>;
	using ImageMultipleRegionType = TexSyn::MipmapMultiIDMap::ImageMultiIDMapType;

	LocallyStationaryTextureSynthesizer();

	void setRegionMap(Ref<Image> regions);
	
	void originsMapToImage(Ref<Image> origins);
	void simplifiedRegionMapToImage(Ref<Image> regionsSimplified);
	
	void invTFilteredToTexture2DArrayAlbedo(Ref<Texture2DArray> invTFilteredRef);
	void invPCAFilteredToTexture2DArrayAlbedo(Ref<Texture2DArray> invPCAFilteredRef);
	
	void computeInvT();
	void computeGaussianExemplar();
	
	void computeExemplarInLocalPCAs();
	void computeInvLocalPCAs();
	
	void setDebugSaves(bool b);
	void test();

protected:
	static void _bind_methods();

private:

	void precomputationsPrefiltering();
	void precomputationsGaussian();
	void precomputationsLocalPCAs();
	void precomputationsLocalPCAsBetter();
	void computeExemplarWithOnlyPCAOfRegion(ImageVectorType &texture, uint64_t region);
	
	ImageVectorType debug_visualizeRegions(const ImageMultipleRegionType &map);

	unsigned int m_nbRegions;
	ImageRegionType m_regionsInt;
	ImageMultipleRegionType m_multiIdMap;
	TexSyn::GaussianTransfer m_gst;
	bool m_debugSaves;
	
	TexSyn::MipmapMultiIDMap m_mipmapMultiIDMap;
	TexSyn::Mipmap m_mipmapExemplar;
	
	ImageVectorType m_invT; //< stores the inverse transfer
	ImageVectorType m_TG; //< stores the Gaussianized exemplar
	
	ImageVectorType m_exemplarPCA; //< stores the exemplar in local PCA spaces
	ImageVectorType m_invPCA; //< stores the inverse local PCAs
	LocalVector<ImageVectorType> m_invPCAPreFiltered; //< stores the pre-filtered inverse local PCAs
};

#define TEXSYN_TESTS
#ifdef TEXSYN_TESTS

bool texsyn_tests();

VARIANT_ENUM_CAST(TextureSynthesizer::TextureTypeFlag);

#endif //ifdef TEXSYN_TESTS

#endif // TEXSYN_H

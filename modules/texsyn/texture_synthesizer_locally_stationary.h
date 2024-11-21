#ifndef LOCALLYSTATIONARYTEXTURESYNTHESIZER_H
#define LOCALLYSTATIONARYTEXTURESYNTHESIZER_H

#include "texture_synthesizer.h"

class TextureSynthesizerLocallyStationary : public TextureSynthesizer
{
	GDCLASS(TextureSynthesizerLocallyStationary, TextureSynthesizer);
	
public:

	enum TransferMode
	{
		PCA_ONLY = 1,
		HISTOGRAM_ONLY = 2,
		FULL = 3
	};

	using ImageRegionType = TexSyn::ImageScalar<int>;
	using PCAType = TexSyn::PCA<float>;
	using ImageMultipleRegionType = TexSyn::MipmapMultiIDMap::ImageMultiIDMapType;

	TextureSynthesizerLocallyStationary();
	
	void setRegionMap(Ref<Image> regions);
	void setMode(TransferMode mode);
	
	void generate();
	
	void originsMapToImage(Ref<Image> origins);
	void simplifiedRegionMapToImage(Ref<Image> regionsSimplified);
	
	void invPCAFilteredToTexture2DArray(Ref<Texture2DArray> invPCAFilteredRef, TextureTypeFlag type);
	void invTFilteredToTexture2DArray(Ref<Texture2DArray> invTFilteredRef, TextureTypeFlag type);
	void invTAndPCAToTexture2DArray(Ref<Texture2DArray> invFilteredRef, TextureTypeFlag type);
	void regionalContributionsToTexture2DArray(Ref<Texture2DArray> regionalContributionsRef);
	void compactContributionsToTexture2DArray(Ref<Texture2DArray> compactContributionsRef);
	void compactContributionsToImage(Ref<Image> compactContributionsRef);
	
	void invTToImage(Ref<Image> invTRef, TextureTypeFlag type);
	void GaussianExemplarToImage(Ref<Image> GaussianExemplarRef, TextureTypeFlag type);
	
	void exemplarInLocalPCAsToImage(Ref<Image> ExemplarPCARef, TextureTypeFlag type);
	void invLocalPCAToImage(Ref<Image> invLocalPCARef, TextureTypeFlag type);
	
	void groundTruthAtlasesTo2DArrayAlbedo(Ref<Texture2DArray> atlasesRef, Ref<Image> originsRef, Ref<Image> regionRef);
	
	void setDebugSaves(bool b);
	void test();

protected:
	static void _bind_methods();

private:

	void generatePCATransfer();
	void generateHistogramTransfer();
	void generateFullTransfer();

	void precomputationsPrefiltering();
	void precomputationsGaussian();
	void precomputationsLocalPCAs();
	void computeExemplarWithOnlyPCAOfRegion(ImageVectorType &texture, uint64_t region, bool saveIntermediateTransfer = true);
	ImageVectorType::VectorType footprintVariance(const ImageVectorType &texture, unsigned int level, uint64_t region);
	
	ImageVectorType debug_visualizeRegions(const ImageMultipleRegionType &map);

	TransferMode m_mode;

	unsigned int m_nbRegions;
	ImageRegionType m_regionsInt;
	ImageMultipleRegionType m_multiIdMap;
	TexSyn::GaussianTransfer m_gst;
	bool m_debugSaves;
	
	TexSyn::MipmapMultiIDMap m_mipmapMultiIDMap;
	TexSyn::Mipmap m_mipmapExemplar;
	
	ImageVectorType m_invT; //< stores the inverse transfer
	ImageVectorType m_GaussianExemplar; //< stores the Gaussianized exemplar
	
	ImageVectorType m_exemplarPCA; //< stores the exemplar in local PCA spaces
	ImageVectorType m_invPCA; //< stores the inverse local PCAs
	ImageVectorType m_origins; //< stores the origins, used to seed the regions
	Vector<ImageVectorType> m_invPCAPreFiltered; //< stores the pre-filtered inverse local PCAs
	Vector<ImageVectorType> m_invTPreFiltered; //< stores the pre-filtered local histogram transfers
	
	Vector<TexSyn::Mipmap> m_regionsContributions; //< stores the contributions of each region
};

VARIANT_ENUM_CAST(TextureSynthesizerLocallyStationary::TransferMode);

#endif // LOCALLYSTATIONARYTEXTURESYNTHESIZER_H

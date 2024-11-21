#include "texture_synthesizer_locally_stationary.h"

TextureSynthesizerLocallyStationary::TextureSynthesizerLocallyStationary() :
TextureSynthesizer(),
m_mode(),
m_nbRegions(0),
m_regionsInt(),
m_multiIdMap(),
m_gst(),
m_debugSaves(false),
m_invT(),
m_GaussianExemplar(),
m_exemplarPCA(),
m_invPCA(),
m_invPCAPreFiltered(),
m_invTPreFiltered(),
m_regionsContributions(),
m_mipmapMultiIDMap(),
m_mipmapExemplar()
{}

void TextureSynthesizerLocallyStationary::setRegionMap(Ref<Image> regions)
{
	ERR_FAIL_COND_MSG(regions.is_null(), "regions must not be null.");

	const int minSize = 256;

	//collecting all ids, creating regionsInt (regions as integers)
	using MapType = HashMap<Color, int>;
	MapType histogramRegions;
	m_regionsInt.init(regions->get_width(), regions->get_height(), true);
	m_nbRegions = 1;
	m_regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
	{
		Color c = regions->get_pixel(x, y);
		if(c.is_equal_approx(Color(1, 1, 1)) || c.is_equal_approx(Color(0, 0, 0)))
		{
			pix = 0;
		}
		else
		{
			MapType::Iterator it = histogramRegions.find(c);
			if(it != histogramRegions.end())
			{
				pix = it->value;
			}
			else
			{
				MapType::Iterator it2 = histogramRegions.insert(c, m_nbRegions);
				++m_nbRegions;
				pix = it2->value;
			}
		}
	});
	
	//Counting region sizes
	std::vector<int> count;
	count.resize(m_nbRegions);
	//Count region areas
	m_regionsInt.for_all_pixels([&] (const ImageRegionType::DataType &pix)
	{
		++count[pix];
	});
	
	//Curate regions that are too small and can cause issue with histogram transfer
	ImageRegionType::DataType regionOffset=0;
	ImageRegionType::DataType lastCuratedRegion=0;
	unsigned int newNbRegions = m_nbRegions;
	for(ImageRegionType::DataType i=0; i<m_nbRegions; ++i)
	{
		m_regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix)
		{
			if(pix == i)
			{
				if(count[pix]<minSize)
				{
					if(lastCuratedRegion != pix && pix != 0)
					{
						lastCuratedRegion = pix;
						++regionOffset;
						--newNbRegions;
					}
					pix = 0;
				}
				else
				{
					pix -= regionOffset;
				}
			}
		});
	}
	m_nbRegions = newNbRegions;
	//Pre-computing the region mask bitmask version
	TexSyn::GaussianTransfer::toMultipleRegions(m_multiIdMap, m_regionsInt);
	print_line(String("Computed a region map with NUM regions.").replace("NUM", String::num_int64(m_nbRegions)));

	if(m_debugSaves)
	{
		TexSyn::ImageScalar<double> regionsOutput;
		regionsOutput.init(m_regionsInt.get_width(), m_regionsInt.get_height(), true);
		regionsOutput.for_all_pixels([&] (double & pix, int x, int y)
		{
			int region = m_regionsInt.get_pixel(x, y);
			pix = float(region)/(m_nbRegions-1);
		});

		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(regionsOutput.get_width(), regionsOutput.get_height(), false, Image::FORMAT_RF);
		regionsOutput.toImage(tmpResultRef, 0);
		tmpResultRef->save_png(DEBUG_FOLDER + "/regionMapInt.png");
	}
}

void TextureSynthesizerLocallyStationary::setMode(TransferMode mode)
{
	m_mode = mode;
}

void TextureSynthesizerLocallyStationary::generate()
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	
	computeImageVector();
	
	if(m_mode == TransferMode::FULL)
	{
		generateFullTransfer();
	}
	else if(m_mode == TransferMode::HISTOGRAM_ONLY)
	{
		generateHistogramTransfer();
	}
	else if(m_mode == TransferMode::PCA_ONLY)
	{
		generatePCATransfer();
	}
	else
	{
		ERR_PRINT("Unknown mode");
	}
}

void TextureSynthesizerLocallyStationary::originsMapToImage(Ref<Image> origins)
{
	ERR_FAIL_COND_MSG(origins.is_null(), "origins must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");

	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(m_origins.get_width(), m_origins.get_height(), false, Image::FORMAT_RGF);
		m_origins.toImageIndexed(tmpResultRef, 0);
		origins->copy_from(tmpResultRef);
	}
}

void TextureSynthesizerLocallyStationary::simplifiedRegionMapToImage(Ref<Image> regionsSimplified)
{
	ERR_FAIL_COND_MSG(regionsSimplified.is_null(), "regionsSimplified must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	TexSyn::ImageScalar<double> regionsOutput;
	regionsOutput.init(m_regionsInt.get_width(), m_regionsInt.get_height(), true);
	regionsOutput.for_all_pixels([&] (double & pix, int x, int y)
	{
		int region = m_regionsInt.get_pixel(x, y);
		pix = float(region)/(m_nbRegions-1);
	});
	
	Ref<Image> tmpResultRef;
	tmpResultRef = Image::create_empty(regionsOutput.get_width(), regionsOutput.get_height(), false, Image::FORMAT_RF);
	regionsOutput.toImage(tmpResultRef, 0);
	regionsSimplified->copy_from(tmpResultRef);
}

void TextureSynthesizerLocallyStationary::invPCAFilteredToTexture2DArray(Ref<Texture2DArray> invPCAFilteredRef, TextureTypeFlag component)
{
	ERR_FAIL_COND_MSG(true, "This function is desactivated.");

	ERR_FAIL_COND_MSG(invPCAFilteredRef.is_null(), "invTFilteredRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(m_invPCAPreFiltered.is_empty(), "computeExemplarInLocalPCAs() or computeInvLocalPCAs() should be called before this function.");
	
	outputImageArrayToComponent(m_invPCAPreFiltered, component, invPCAFilteredRef);
	
	if(m_debugSaves)
	{
		for(int i=0; i<invPCAFilteredRef->get_layers(); ++i)
		{
			invPCAFilteredRef->get_layer_data(i)->save_png((DEBUG_FOLDER + "/invPCARef_num.png").replace("num", String::num_int64(i)));
		}
	}
}

void TextureSynthesizerLocallyStationary::invTFilteredToTexture2DArray(Ref<Texture2DArray> invTFilteredRef, TextureTypeFlag component)
{
	ERR_FAIL_COND_MSG(invTFilteredRef.is_null(), "invTFilteredRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(!m_exemplar.is_initialized(), "computeInvT() or computeGaussianExemplar() should be called before this function.");
	//computing the filtered region map

	outputImageArrayToComponent(m_invTPreFiltered, component, invTFilteredRef);
	
	if(m_debugSaves)
	{
		for(int i=0; i<invTFilteredRef->get_layers(); ++i)
		{
			invTFilteredRef->get_layer_data(i)->save_png(String(DEBUG_FOLDER + "/invTFiltered_num.png").replace("num", String::num_int64(i)));
		}
	}
}

void TextureSynthesizerLocallyStationary::regionalContributionsToTexture2DArray(Ref<Texture2DArray> regionalContributionsRef)
{
	ERR_FAIL_COND_MSG(regionalContributionsRef.is_null(), "regionalContributionsRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(m_regionsContributions.is_empty(), "computeExemplarInLocalPCAs() or computeInvLocalPCAs() should be called before this function.");
	
	Vector<Ref<Image>> regionalContributionsVectorRef;
	regionalContributionsVectorRef.resize(m_nbRegions);
	
	for(unsigned int i=0; i<m_regionsContributions.size(); ++i)
	{
		const TexSyn::Mipmap &mipmap = m_regionsContributions[i];
		const ImageVectorType &image = mipmap.mipmap(0);
		Ref<Image> tmpResultRef = Image::create_empty(image.get_width(), image.get_height(), true, Image::FORMAT_RF);
		image.toImage(tmpResultRef);
		regionalContributionsVectorRef.write[i].instantiate();
		regionalContributionsVectorRef.write[i]->copy_from(tmpResultRef);
		//Optionnal previsualisation save
		if(m_debugSaves)
		{
			for(int k=0; k<1; ++k)
			//for(int k=0; k<mipmap.nbMaps(); ++k)
			{
				const ImageVectorType &map = mipmap.mipmap(k);
				Ref<Image> tmpResultRef = Image::create_empty(map.get_width(), map.get_height(), true, Image::FORMAT_RF);
				map.toImage(tmpResultRef);
				tmpResultRef->save_png((DEBUG_FOLDER + "/contribution_num_mipmap_k.png").replace("num", String::num_int64(i)).replace("k", String::num_int64(k)));
			}
		}
	}
	regionalContributionsRef->create_from_images(regionalContributionsVectorRef);
}

void TextureSynthesizerLocallyStationary::compactContributionsToTexture2DArray(Ref<Texture2DArray> compactContributionsRef)
{
	ERR_FAIL_COND_MSG(compactContributionsRef.is_null(), "regionalContributionsRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(m_regionsContributions.is_empty(), "computeExemplarInLocalPCAs() or computeInvLocalPCAs() should be called before this function.");
	
	Vector<Ref<Image>> compactContributionsVectorRef;
	compactContributionsVectorRef.resize(m_mipmapExemplar.nbMaps());
	
	//irreversible: upsize all mipmaps of the contributions (heavy memory load)
	for(unsigned int i=0; i<m_regionsContributions.size(); ++i)
	{
		m_regionsContributions.write[i].upsizeMipmap();
	}
	
	for(int k=0; k<m_mipmapExemplar.nbMaps(); ++k)
	{
		ImageVectorType contributions;
		contributions.init(m_exemplar.get_width(), m_exemplar.get_height(), 2, true);
		
		//First dimension stores the contribution of the background
		contributions.get_image(0).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
		{
			const TexSyn::Mipmap &map = m_regionsContributions[0];
			const ImageScalarType &image = map.mipmap(k).get_image(0);
			pix = image.get_pixel(x, y);
		});
		contributions.get_image(1).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
		{
			int region = m_regionsInt.get_pixel(x, y);
			if(region != 0)
			{
				const TexSyn::Mipmap &map = m_regionsContributions[region];
				const ImageScalarType &image = map.mipmap(k).get_image(0);
				pix = image.get_pixel(x, y);
			}
		});
		
		Ref<Image> tmpResultRef = Image::create_empty(contributions.get_width(), contributions.get_height(), false, Image::FORMAT_RGF);
		contributions.toImage(tmpResultRef);
		compactContributionsVectorRef.write[k].instantiate();
		compactContributionsVectorRef.write[k]->copy_from(tmpResultRef);
	}
	compactContributionsRef->create_from_images(compactContributionsVectorRef);
}

void TextureSynthesizerLocallyStationary::compactContributionsToImage(Ref<Image> compactContributionsRef)
{
	ERR_FAIL_COND_MSG(compactContributionsRef.is_null(), "regionalContributionsRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(m_regionsContributions.is_empty(), "computeExemplarInLocalPCAs() or computeInvLocalPCAs() should be called before this function.");
	
	//This version computes a mipmap instead of an array of images.
	//The first channel contains the dominant region ID and the two others contain the contributions.
	
	TexSyn::Mipmap mipmap;
	ImageVectorType contributions;
	contributions.init(m_exemplar.get_width(), m_exemplar.get_height(), 3, true);
	mipmap.setTexture(contributions);
	mipmap.computeMipmap();
	
	for(int k=0; k<m_mipmapExemplar.nbMaps(); ++k)
	{
		ImageVectorType &contributions = mipmap.mipmap(k);
		
		//First dimension stores the contribution of the background
		contributions.get_image(1).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
		{
			const TexSyn::Mipmap &map = m_regionsContributions[0];
			const ImageScalarType &image = map.mipmap(k).get_image(0);
			pix = image.get_pixel(x, y);
		});
		contributions.get_image(2).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
		{
			int maxRegion = 0;
			if(k == 0) //small optimisation at highest resolution
			{
				maxRegion = m_regionsInt.get_pixel(x, y);
				if(maxRegion != 0)
				{
					pix = 1.0;
				}
			}
			else
			{
				for(unsigned int region=1; region<m_nbRegions; ++region)
				{
					const TexSyn::Mipmap &map = m_regionsContributions[region];
					const ImageScalarType &image = map.mipmap(k).get_image(0);
					float contribution = image.get_pixel(x, y);
					if(contribution > pix)
					{
						pix = fmax(pix, image.get_pixel(x, y));
						maxRegion = region;
					}
				}
			}
			contributions.get_image(0).set_pixel(x, y, float(maxRegion)/(m_nbRegions-1));
		});
	}
	mipmap.toImage(compactContributionsRef, Image::FORMAT_RGBF);
	
	if(m_debugSaves)
	{
		for(int k=0; k<mipmap.nbMaps(); ++k)
		{
			const ImageVectorType &map = mipmap.mipmap(k);
			ImageVectorType mapVisualisation;
			mapVisualisation.init(map.get_width(), map.get_height(), map.get_nbDimensions());
			mapVisualisation.get_image(0).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
			{
				pix = map.get_image(2).get_pixel(x, y);
			});
			mapVisualisation.get_image(1).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
			{
				pix = map.get_image(1).get_pixel(x, y);
			});
			mapVisualisation.get_image(2).for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
			{
				pix = 1.0 - mapVisualisation.get_image(0).get_pixel(x, y) - mapVisualisation.get_image(1).get_pixel(x, y);
			});
			Ref<Image> tmpResultRef = Image::create_empty(mapVisualisation.get_width(), mapVisualisation.get_height(), true, Image::FORMAT_RGBF);
			mapVisualisation.toImage(tmpResultRef);
			tmpResultRef->save_png((DEBUG_FOLDER + "/compactContributions_mipmap_k.png").replace("k", String::num_int64(k)));
		}
	}
}

void TextureSynthesizerLocallyStationary::invTToImage(Ref<Image> invTRef, TextureTypeFlag type)
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(!m_invT.is_initialized(), "invT must be initialized.");
	
	outputImageToComponent(m_invT, type, invTRef);
}

void TextureSynthesizerLocallyStationary::GaussianExemplarToImage(Ref<Image> GaussianExemplarRef, TextureTypeFlag type)
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(!m_GaussianExemplar.is_initialized(), "GaussianExemplar must be initialized.");
	
	outputImageToComponent(m_GaussianExemplar, type, GaussianExemplarRef);
}

void TextureSynthesizerLocallyStationary::exemplarInLocalPCAsToImage(Ref<Image> ExemplarPCARef, TextureTypeFlag type)
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(!m_exemplarPCA.is_initialized(), "exemplarPCA must be initialized.");
	
	//In order to save in .png, add 0.5
	ImageVectorType output(m_exemplarPCA);
	output.for_all_images([&] (ImageVectorType::ImageScalarType &image, unsigned int d)
	{
		image += 0.5;
	});
	outputImageToComponent(output, type, ExemplarPCARef);
}

void TextureSynthesizerLocallyStationary::invLocalPCAToImage(Ref<Image> invPCARef, TextureTypeFlag type)
{	
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(!m_invPCA.is_initialized(), "invLocalPCA must be initialized.");
	
	outputImageToComponent(m_invPCA, type, invPCARef);
}

void TextureSynthesizerLocallyStationary::groundTruthAtlasesTo2DArrayAlbedo(Ref<Texture2DArray> atlasesRef, Ref<Image> originsRef, Ref<Image> regionRef)
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(originsRef.is_null(), "origins must not be null.");
	ERR_FAIL_COND_MSG(regionRef.is_null(), "region must not be null.");
	
	Ref<Image> tmpResultRef;
	
	//Exporting region map, using custom Image class code
	ImageVectorType imageRegionReadable;
	using BitVector = TexSyn::BitVector;
	const TexSyn::ImageScalar<BitVector> &multiIDMap = m_mipmapMultiIDMap.mipmap(0);
	imageRegionReadable.init(multiIDMap.get_width(), multiIDMap.get_height(), 4, false);
	for(int y=0; y<multiIDMap.get_height(); ++y)
	{
		for(int x=0; x<multiIDMap.get_width(); ++x)
		{
			BitVector bit = multiIDMap.get_pixel(x, y);
			
			uint32_t pix32 = bit.lo & 0xffffffff;
			ImageVectorType::DataType &pix3 = imageRegionReadable.get_pixelRef(x, y, 3);
			reinterpret_cast<uint32_t &>(pix3) = pix32;
//			print_line(String("a: bit at xx, yy: val").replace("val", String::num_uint64(pix32)).replace("xx", String::num_int64(x)).replace("yy", String::num_int64(y)));

			pix32 = (bit.lo >> 32) & 0xffffffff;
			ImageVectorType::DataType &pix2 = imageRegionReadable.get_pixelRef(x, y, 2);
			reinterpret_cast<uint32_t &>(pix2) = pix32;
//			print_line(String("b: bit at xx, yy: val").replace("val", String::num_uint64(pix32)).replace("xx", String::num_int64(x)).replace("yy", String::num_int64(y)));
			
			pix32 = bit.hi & 0xffffffff;
			ImageVectorType::DataType &pix1 = imageRegionReadable.get_pixelRef(x, y, 1);
			reinterpret_cast<uint32_t &>(pix1) = pix32;
//			print_line(String("g: bit at xx, yy: val").replace("val", String::num_uint64(pix32)).replace("xx", String::num_int64(x)).replace("yy", String::num_int64(y)));
		}
	}
	tmpResultRef = Image::create_empty(multiIDMap.get_width(), multiIDMap.get_height(), true, Image::FORMAT_RGBAF);
	tmpResultRef->mark_as_IDMap(); //custom code
	imageRegionReadable.toImage(tmpResultRef);
	tmpResultRef->generate_mipmaps(false);
	regionRef->copy_from(tmpResultRef);
}

void TextureSynthesizerLocallyStationary::precomputationsPrefiltering()
{
	if(m_mipmapMultiIDMap.nbMaps()>0)
		return;
	m_mipmapMultiIDMap.setIDMap(m_multiIdMap);
	m_mipmapMultiIDMap.computeMipmap();

	//computing the exemplar mipmap
	m_mipmapExemplar.setTexture(m_exemplar);
	m_mipmapExemplar.computeMipmap();
	
	if(m_debugSaves)
	{
		for(int i=0; i<m_mipmapMultiIDMap.nbMaps(); ++i)
		{
			const ImageMultipleRegionType &map = m_mipmapMultiIDMap.mipmap(i);
			ImageVectorType mapVisu = debug_visualizeRegions(map);
			Ref<Image> tmpResultRef = Image::create_empty(map.get_width(), map.get_height(), false, Image::FORMAT_RGBF);
			mapVisu.toImageIndexed(tmpResultRef, 0);
			tmpResultRef->save_png((DEBUG_FOLDER + "/mipmapRegions_num.png").replace("num", String::num_int64(i)));
		}
		for(int i=0; i<m_mipmapExemplar.nbMaps(); ++i)
		{
			const ImageVectorType &map = m_mipmapExemplar.mipmap(i);
			Ref<Image> tmpResultRef = Image::create_empty(map.get_width(), map.get_height(), false, Image::FORMAT_RGBF);
			map.toImageIndexed(tmpResultRef, 0);
			tmpResultRef->save_png((DEBUG_FOLDER + "/mipmapExemplar_num.png").replace("num", String::num_int64(i)));
		}
	}
	
	//Computing the contribution of each region
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		ImageVectorType contribution;
		contribution.init(m_exemplar.get_width(), m_exemplar.get_height(), 1, true);
		ImageScalarType &contributionScalar = contribution.get_image(0);
		contributionScalar.for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
		{
			ImageMultipleRegionType::DataType pixRegion = m_multiIdMap.get_pixel(x, y);
			if((pixRegion & uint64_t(i)).toBool())
			{
				pix = 1.0;
			}
		});
		TexSyn::Mipmap mipmap;
		mipmap.setTexture(contribution);
		mipmap.computeMipmap();
		m_regionsContributions.push_back(mipmap);
	}
	
	if(m_debugSaves)
	{
		for(unsigned int i=0; i<m_nbRegions; ++i)
		{
			const ImageVectorType &map = m_regionsContributions[i].mipmap(0);
			Ref<Image> tmpResultRef = Image::create_empty(map.get_width(), map.get_height(), false, Image::FORMAT_RF);
			map.toImageIndexed(tmpResultRef, 0);
			tmpResultRef->save_png((DEBUG_FOLDER + "/regionsContributions_num.png").replace("num", String::num_int64(i)));
		}
	}
}

void TextureSynthesizerLocallyStationary::precomputationsGaussian()
{
	//Pre-computation of invT
	const int invTSize = 128;
	if(m_exemplarPCA.is_initialized())
	{
		m_exemplarPCA.for_all_images([&] (ImageVectorType::ImageScalarType &image, unsigned int d)
		{
			image += 0.5;
		});
		m_invT.init(invTSize, m_nbRegions, m_exemplarPCA.get_nbDimensions(), true);
		m_gst.computeinvTRegions(m_exemplarPCA, m_regionsInt, m_invT);
	
		m_GaussianExemplar.init(m_exemplarPCA.get_width(), m_exemplarPCA.get_height(), m_exemplarPCA.get_nbDimensions(), true);
		m_gst.computeTinputRegions(m_exemplarPCA, m_regionsInt, m_GaussianExemplar, false, false);
	}
	else
	{
		m_invT.init(invTSize, m_nbRegions, m_exemplar.get_nbDimensions(), true);
		m_gst.computeinvTRegions(m_exemplar, m_regionsInt, m_invT);
	
		m_GaussianExemplar.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
		m_gst.computeTinputRegions(m_exemplar, m_regionsInt, m_GaussianExemplar, false, false);
	}
}

void TextureSynthesizerLocallyStationary::precomputationsLocalPCAs()
{
#define ADDGLOBALPCA
	precomputationsPrefiltering();
	m_mipmapMultiIDMap.upsizeMipmap();
	m_mipmapExemplar.upsizeMipmap();
	//computing local PCAs, projected exemplar, and packing inverse PCA infos
	m_exemplarPCA.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	
	unsigned int regionOffset = 0;
#ifdef ADDGLOBALPCA
	regionOffset = 1;
#endif
	m_invPCA.init(1+m_exemplar.get_nbDimensions(), m_nbRegions+regionOffset, m_exemplar.get_nbDimensions(), true);
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		PCAType localPCA(m_exemplar, m_multiIdMap, uint64_t(i));
		localPCA.computePCA();
		localPCA.computeProjection();
		localPCA.project(m_exemplarPCA);
		PCAType::MatrixType localEigenVectors = localPCA.get_eigenVectors().transpose();
		PCAType::VectorType localMean = localPCA.get_mean();
		//filling invPCA: at x=0, mean, and then eigen vectors
		for(unsigned int d=0; d<m_exemplar.get_nbDimensions(); ++d)
		{
			m_invPCA.set_pixel(0, i+regionOffset, d, localMean[d]);
			for(int r=0; r<localEigenVectors.rows(); ++r)
			{
				m_invPCA.set_pixel(r+1, i+regionOffset, d, localEigenVectors(r, d));
			}
		}
	}
	
#ifdef ADDGLOBALPCA
	//Sneaky insertion of the global PCA in the first position of the pack
	ImageMultipleRegionType foregroundRegions;
	foregroundRegions.init(m_exemplar.get_width(), m_exemplar.get_width(), true);
	foregroundRegions.for_all_pixels([&] (ImageMultipleRegionType::DataType &pix, int x, int y)
	{
		ImageRegionType::DataType region = m_regionsInt.get_pixel(x, y);
		pix |= region == 0 ? 0 : 1;
	});
	PCAType globalPCA(m_exemplar, foregroundRegions, 1);
	globalPCA.computePCA();
	PCAType::MatrixType localEigenVectors = globalPCA.get_eigenVectors().transpose();
	PCAType::VectorType localMean = globalPCA.get_mean();
	for(unsigned int d=0; d<m_exemplar.get_nbDimensions(); ++d)
	{
		m_invPCA.set_pixel(0, 0, d, localMean[d]);
		for(int r=0; r<localEigenVectors.rows(); ++r)
		{
			m_invPCA.set_pixel(r+1, 0, d, localEigenVectors(r, d));
		}
	}
#endif

	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		ImageVectorType pcaTextureRegion;
		computeExemplarWithOnlyPCAOfRegion(pcaTextureRegion, uint64_t(i));
	}
	
	//Computing inverse PCAs, but prefiltered
	// m_invPCAPreFiltered.resize(m_mipmapExemplar.nbMaps());
	// LocalVector<LocalVector<PCAType>> localPCAsPreFiltered;
	// localPCAsPreFiltered.resize(m_mipmapExemplar.nbMaps());
	

	// for(unsigned int k=0; k<localPCAsPreFiltered.size(); ++k)
	// {
	// 	LocalVector<PCAType> &localPCAs = localPCAsPreFiltered[k];
	// 	const ImageVectorType &exemplar = m_mipmapExemplar.mipmap(k);
	// 	const ImageMultipleRegionType &multipleRegions = m_mipmapMultiIDMap.mipmap(k);
	// 	ImageVectorType &invPCA = m_invPCAPreFiltered.write[k];
	// 	invPCA.init(1+exemplar.get_nbDimensions(), m_nbRegions, exemplar.get_nbDimensions(), true);
	// 	for(unsigned int i=0; i<m_nbRegions; ++i)
	// 	{
	// 		localPCAs.push_back(PCAType(exemplar, multipleRegions, uint64_t(i), false));
	// 		PCAType &localPCA = localPCAs[i];
	// 		localPCA.computePCA();
	// 		PCAType::MatrixType localEigenVectors = localPCA.get_eigenVectors().transpose();
	// 		PCAType::VectorType localMean = localPCA.get_mean();
	// 		//filling invPCA: at x=0, mean, and then eigen vectors
	// 		for(unsigned int d=0; d<exemplar.get_nbDimensions(); ++d)
	// 		{
	// 			invPCA.set_pixel(0, i, d, localMean[d]);
	// 			for(int r=0; r<localEigenVectors.rows(); ++r)
	// 			{
	// 				invPCA.set_pixel(r+1, i, d, localEigenVectors(r, d));
	// 			}
	// 		}
	// 	}
	// }
	
	//Initialize origins
	m_origins.init(m_nbRegions, 1, 2);
	for(unsigned int otherID=0; otherID<m_nbRegions; ++otherID)
	{
		float dx = 0.0, dy = 0.0;
		int maxX=0, maxY=0, minX=m_regionsInt.get_width()-1, minY=m_regionsInt.get_height()-1;
		m_regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
		{
			if(pix == otherID)
			{
				maxX = std::max(maxX, x);
				maxY = std::max(maxY, y);
				minX = std::min(minX, x);
				minY = std::min(minY, y);
			}
		});
		if(maxX == m_regionsInt.get_width()-1 && minX == 0)
		{
			dx = 0.5;
		}
		if(maxY == m_regionsInt.get_width()-1 && minY == 0)
		{
			dy = 0.5;
		}
		m_origins.set_pixel(otherID, 0, 0, dx);
		m_origins.set_pixel(otherID, 0, 1, dy);
	}
	
	if(!m_debugSaves)
		return;

	//Testing by simulating the GPU process
	ImageVectorType outputPCA;
	outputPCA.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	for(int x=0; x<outputPCA.get_width(); ++x)
	{
		for(int y=0; y<outputPCA.get_height(); ++y)
		{
			int region = m_regionsInt.get_pixel(x, y);
			ImageVectorType::VectorType mean = m_invPCA.get_pixel(0, region+1);
			ImageVectorType::VectorType p1 = m_invPCA.get_pixel(1, region+1);
			ImageVectorType::VectorType p2 = m_invPCA.get_pixel(2, region+1);
			ImageVectorType::VectorType p3 = m_invPCA.get_pixel(3, region+1);
			ImageVectorType::VectorType inputPCA = m_exemplarPCA.get_pixel(x, y);
			ImageVectorType::VectorType v = mean;
			v.write[0] += inputPCA[0]*p1[0] + inputPCA[1]*p2[0] + inputPCA[2]*p3[0];
			v.write[1] += inputPCA[0]*p1[1] + inputPCA[1]*p2[1] + inputPCA[2]*p3[1];
			v.write[2] += inputPCA[0]*p1[2] + inputPCA[1]*p2[2] + inputPCA[2]*p3[2];
			outputPCA.set_pixel(x, y, v);
		}
	}
	Ref<Image> tmpResultRef;
	tmpResultRef = Image::create_empty(outputPCA.get_width(), outputPCA.get_height(), false, Image::FORMAT_RGBF);
	outputPCA.toImageIndexed(tmpResultRef, 0);
	tmpResultRef->save_png("test.png");
	
	//Testing for real
	outputPCA.init(m_exemplar.get_width()*2, m_exemplar.get_height()*2, m_exemplar.get_nbDimensions(), true);
	RandomNumberGenerator rng;
	for(int x=0; x<outputPCA.get_width(); ++x)
	{
		for(int y=0; y<outputPCA.get_height(); ++y)
		{
			int region = m_regionsInt.get_pixel(x%m_regionsInt.get_width(), y%m_regionsInt.get_height());
			float orX = m_origins.get_pixel(region, 0, 0);
			float orY = m_origins.get_pixel(region, 0, 1);
			int seedX = int(x -  orX*m_exemplar.get_width())/m_regionsInt.get_width();
			int seedY = int(y -  orY*m_exemplar.get_height())/m_regionsInt.get_height();
			int seed = (region << 20) | (seedX << 10) | seedY;
			rng.set_seed(seed);
			int newRegion;
			if(region != 0)
			{
				newRegion = rng.randi_range(1, m_nbRegions-1);
			}
			else
			{
				newRegion = 0;
			}
			ImageVectorType::VectorType mean = m_invPCA.get_pixel(0, newRegion+1);
			ImageVectorType::VectorType p1 = m_invPCA.get_pixel(1, newRegion+1);
			ImageVectorType::VectorType p2 = m_invPCA.get_pixel(2, newRegion+1);
			ImageVectorType::VectorType p3 = m_invPCA.get_pixel(3, newRegion+1);
			ImageVectorType::VectorType inputPCA = m_exemplarPCA.get_pixel(x%m_regionsInt.get_width(), y%m_regionsInt.get_height());
			ImageVectorType::VectorType v = mean;
			v.write[0] += inputPCA[0]*p1[0] + inputPCA[1]*p2[0] + inputPCA[2]*p3[0];
			v.write[1] += inputPCA[0]*p1[1] + inputPCA[1]*p2[1] + inputPCA[2]*p3[1];
			v.write[2] += inputPCA[0]*p1[2] + inputPCA[1]*p2[2] + inputPCA[2]*p3[2];
			outputPCA.set_pixel(x, y, v);
		}
	}
	tmpResultRef = Image::create_empty(outputPCA.get_width(), outputPCA.get_height(), false, Image::FORMAT_RGBF);
	outputPCA.toImageIndexed(tmpResultRef, 0);
	tmpResultRef->save_png("testRng.png");
	
	//
}

void TextureSynthesizerLocallyStationary::computeExemplarWithOnlyPCAOfRegion(ImageVectorType &texture, uint64_t region, bool saveIntermediateTransfer)
{
	ERR_FAIL_COND_MSG(!m_exemplarPCA.is_initialized(), "Exemplar in PCA space must have been computed before calling this function.");
	texture.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	for(int x=0; x<texture.get_width(); ++x)
	{
		for(int y=0; y<texture.get_height(); ++y)
		{
			//Optionnal : do not transfer texels marked as region 0
			//int adjustedRegion = m_regionsInt.get_pixel(x, y) == 0 ? 0 : region;
			int adjustedRegion = region;
			ImageVectorType::VectorType inputPCA;
			
			if(m_GaussianExemplar.is_initialized())
			{
				ImageVectorType::VectorType input = m_GaussianExemplar.get_pixel(x, y);
				int r, g, b;
				r = MAX(0, MIN(input[0]*(m_invT.get_width()-1), m_invT.get_width()-1));
				g = MAX(0, MIN(input[1]*(m_invT.get_width()-1), m_invT.get_width()-1));
				b = MAX(0, MIN(input[2]*(m_invT.get_width()-1), m_invT.get_width()-1));
				inputPCA.resize(3);
				inputPCA.write[0] = m_invT.get_pixel(r, adjustedRegion, 0) - 0.5;
				inputPCA.write[1] = m_invT.get_pixel(g, adjustedRegion, 1) - 0.5;
				inputPCA.write[2] = m_invT.get_pixel(b, adjustedRegion, 2) - 0.5;
			}
			else
			{
				inputPCA = m_exemplarPCA.get_pixel(x, y);
			}
			if(m_GaussianExemplar.is_initialized() && saveIntermediateTransfer)
			{
				inputPCA.write[0] += 0.5;
				inputPCA.write[1] += 0.5;
				inputPCA.write[2] += 0.5;
				texture.set_pixel(x, y, inputPCA);
			}
			else
			{
				ImageVectorType::VectorType mean = m_invPCA.get_pixel(0, adjustedRegion+1);
				ImageVectorType::VectorType p1 = m_invPCA.get_pixel(1, adjustedRegion+1);
				ImageVectorType::VectorType p2 = m_invPCA.get_pixel(2, adjustedRegion+1);
				ImageVectorType::VectorType p3 = m_invPCA.get_pixel(3, adjustedRegion+1);
				ImageVectorType::VectorType v = mean;
				v.write[0] += inputPCA[0]*p1[0] + inputPCA[1]*p2[0] + inputPCA[2]*p3[0];
				v.write[1] += inputPCA[0]*p1[1] + inputPCA[1]*p2[1] + inputPCA[2]*p3[1];
				v.write[2] += inputPCA[0]*p1[2] + inputPCA[1]*p2[2] + inputPCA[2]*p3[2];
				texture.set_pixel(x, y, v);
			}
		}
	}
	if(m_debugSaves && !saveIntermediateTransfer)
	{
		Ref<Image> tmpResultRef = Image::create_empty(texture.get_width(), texture.get_height(), false, Image::FORMAT_RGBF);
		texture.toImageIndexed(tmpResultRef, 0);
		tmpResultRef->save_png((DEBUG_FOLDER + "/pcaFullTransfer_num.png").replace("num", String::num_int64(region)));
	}
	else if(m_debugSaves && saveIntermediateTransfer)
	{
		Ref<Image> tmpResultRef = Image::create_empty(texture.get_width(), texture.get_height(), false, Image::FORMAT_RGBF);
		texture.toImageIndexed(tmpResultRef, 0);
		tmpResultRef->save_png((DEBUG_FOLDER + "/pcaHalfTransfer_num.png").replace("num", String::num_int64(region)));
	}
}

TextureSynthesizerLocallyStationary::ImageVectorType::VectorType TextureSynthesizerLocallyStationary::footprintVariance(const ImageVectorType &texture, unsigned int level, uint64_t region)
{
	//This function requires the texture to be a power of 2 square, otherwise you need to replace y+= and x+= (and you do not want that)
	//I need to replace what the VectorType is. It is too inconvenient.
	int windowWidth = int(pow(2.0f, float(level)));
	ImageVectorType::VectorType averageVariance;
	averageVariance.resize(texture.get_nbDimensions());
	for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
	{
		averageVariance.write[d] = 0.0f;
	}
	int totalNbHits = 0;
	for(int y=0; y<texture.get_height(); y+=windowWidth)
	{
		for(int x=0; x<texture.get_width(); x+=windowWidth)
		{
			ImageVectorType::VectorType m;
			ImageVectorType::VectorType v;
			m.resize(texture.get_nbDimensions());
			v.resize(texture.get_nbDimensions());
			for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
			{
				m.write[d] = 0.0f;
				v.write[d] = 0.0f;
			}
			unsigned int nbHits = 0;
			//computing the mean over the footprint
			for(int y2=0; y2<windowWidth; ++y2)
			{
				for(int x2=0; x2<windowWidth; ++x2)
				{
					int xM = (x + x2)%texture.get_width();
					int yM = (y + y2)%texture.get_height();
					int regionPix = m_regionsInt.get_pixel(xM, yM);
					if(regionPix == region)
					{
						++nbHits;
						for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
						{
							m.write[d] += texture.get_pixel(xM, yM, d);
						}
					}
				}
			}
			for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
			{
				m.write[d] /= nbHits;
			}
			if(nbHits>=2)
			{
				totalNbHits += nbHits;
				//computing the variance over the footprint, using the mean
				for(int y2=0; y2<windowWidth; ++y2)
				{
					for(int x2=0; x2<windowWidth; ++x2)
					{
						int xM = (x + x2)%texture.get_width();
						int yM = (y + y2)%texture.get_height();
						int regionPix = m_regionsInt.get_pixel(xM, yM);
						if(regionPix == region)
						{
							for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
							{
								float localVariance = texture.get_pixel(xM, yM, d) - m[d];
								v.write[d] += localVariance*localVariance;
							}
						}
					}
				}
				for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
				{
					averageVariance.write[d] += v.write[d]*nbHits;
				}
			}
		}
	}
	if(totalNbHits > 0)
	{
		for(unsigned int d=0; d<texture.get_nbDimensions(); ++d)
		{
			averageVariance.write[d] /= (totalNbHits*totalNbHits);
		}
	}
	String debugString = String("Average variance of region XX at level LL is (NUM1, NUM2, NUM3)");
	debugString = debugString.replace("XX", String::num_int64(region));
	debugString = debugString.replace("LL", String::num_int64(level));
	debugString = debugString.replace("NUM1", String::num(averageVariance[0], 3));
	debugString = debugString.replace("NUM2", String::num(averageVariance[1], 3));
	debugString = debugString.replace("NUM3", String::num(averageVariance[2], 3));
	//print_line(debugString);
	return averageVariance;
}

TextureSynthesizerLocallyStationary::ImageVectorType TextureSynthesizerLocallyStationary::debug_visualizeRegions(const ImageMultipleRegionType &map)
{
	Ref<Image> tmpResultRef;
	ImageVectorType mapVisualization;
	RandomNumberGenerator rng;
	mapVisualization.init(map.get_width(), map.get_height(), 3, true);
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		mapVisualization.for_all_images([&] (ImageScalarType &image, unsigned int d)
		{
			image.for_all_pixels([&] (ImageScalarType::DataType &pix, int x, int y)
			{
				ImageMultipleRegionType::DataType region = map.get_pixel(x, y);
				if((region & i).toBool())
				{
					if(pix != 0)
					{
						//several regions on the same pixel
						pix = 1;
					}
					else
					{
						rng.set_seed(uint64_t(i * 3 + d));
						pix = rng.randfn(0.5, 0.8);
					}
				}
			});
		});
	}
	return mapVisualization;
}

void TextureSynthesizerLocallyStationary::setDebugSaves(bool b)
{
	m_debugSaves = b;
}

void TextureSynthesizerLocallyStationary::generatePCATransfer()
{
	precomputationsPrefiltering();
	precomputationsLocalPCAs();
}

void TextureSynthesizerLocallyStationary::generateHistogramTransfer()
{
	precomputationsPrefiltering();
	precomputationsGaussian();
	
	//Compute inverse histogram transfer array
	m_invTPreFiltered.resize(m_mipmapExemplar.nbMaps());
	for(unsigned int i=0; i<m_mipmapExemplar.nbMaps(); ++i)
	{
		ImageVectorType &invT = m_invTPreFiltered.write[i];
		invT.init(128, m_nbRegions, m_mipmapExemplar.mipmap(i).get_nbDimensions(), true);
		m_gst.computeinvTMultipleRegions(m_mipmapExemplar.mipmap(i), m_mipmapMultiIDMap.mipmap(i), invT, false);
	}
}

void TextureSynthesizerLocallyStationary::generateFullTransfer()
{
	precomputationsPrefiltering();
	precomputationsLocalPCAs();
	precomputationsGaussian();
	
	m_invTPreFiltered.resize(m_mipmapMultiIDMap.nbMaps());
	m_invTPreFiltered.write[0] = m_invT;
	
	for(int i=1; i<m_invTPreFiltered.size(); ++i)
	{
		const ImageVectorType &firstInvT = m_invTPreFiltered[0];
		//copy the histogram and apply a Gaussian kernel to it
		m_invTPreFiltered.write[i] = firstInvT;
		ImageVectorType &currentInvT = m_invTPreFiltered.write[i];
		int windowWidth = int(pow(2.0f, float(i)));
		
		for(int y=0; y<currentInvT.get_height(); ++y)
		{
			//Computing the average variance over the size of the window
			ImageVectorType::VectorType variance = footprintVariance(m_GaussianExemplar, i, uint64_t(y));
			for(unsigned int d=0; d<m_GaussianExemplar.get_nbDimensions(); ++d)
			{
				m_gst.prefilterLUT(currentInvT, variance[d], uint64_t(y), d);
			}
		}
	}
	
	if(!m_debugSaves)
	{
		return;
	}
	
	//All of the rest is merely for debugging purposes
	
	//Test inverse histogram transfer + inverse PCA once
	ImageVectorType outputTest;
	outputTest.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	for(int x=0; x<outputTest.get_width(); ++x)
	{
		for(int y=0; y<outputTest.get_height(); ++y)
		{
			int region = m_regionsInt.get_pixel(x, y);
			srand(region+10);
			int newRegion = rand()%m_nbRegions;
			
			ImageVectorType::VectorType input = m_GaussianExemplar.get_pixel(x, y);
			int r, g, b;
			r = MAX(0, MIN(input[0]*(m_invT.get_width()-1), m_invT.get_width()-1));
			g = MAX(0, MIN(input[1]*(m_invT.get_width()-1), m_invT.get_width()-1));
			b = MAX(0, MIN(input[2]*(m_invT.get_width()-1), m_invT.get_width()-1));
			ImageVectorType::VectorType inputPCA;
			inputPCA.resize(3);
			inputPCA.write[0] = m_invT.get_pixel(r, newRegion, 0);
			inputPCA.write[1] = m_invT.get_pixel(g, newRegion, 1);
			inputPCA.write[2] = m_invT.get_pixel(b, newRegion, 2);
			outputTest.set_pixel(x, y, inputPCA);
		}
	}
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(outputTest.get_width(), outputTest.get_height(), false, Image::FORMAT_RGBF);
		outputTest.toImageIndexed(tmpResultRef, 0);
		tmpResultRef->save_png("testFullInversePCASpace.png");
	}
	for(int x=0; x<outputTest.get_width(); ++x)
	{
		for(int y=0; y<outputTest.get_height(); ++y)
		{
			int region = m_regionsInt.get_pixel(x, y);
			srand(region+10);
			int newRegion = rand()%m_nbRegions;
			
			ImageVectorType::VectorType inputPCA = outputTest.get_pixel(x, y);
			for(int d=0; d<inputPCA.size(); ++d)
			{
				inputPCA.write[d] -= 0.5f;
			}
			
			ImageVectorType::VectorType mean = m_invPCA.get_pixel(0, newRegion+1);
			ImageVectorType::VectorType p1 = m_invPCA.get_pixel(1, newRegion+1);
			ImageVectorType::VectorType p2 = m_invPCA.get_pixel(2, newRegion+1);
			ImageVectorType::VectorType p3 = m_invPCA.get_pixel(3, newRegion+1);
			
			ImageVectorType::VectorType v = mean;
			v.write[0] += inputPCA[0]*p1[0] + inputPCA[1]*p2[0] + inputPCA[2]*p3[0];
			v.write[1] += inputPCA[0]*p1[1] + inputPCA[1]*p2[1] + inputPCA[2]*p3[1];
			v.write[2] += inputPCA[0]*p1[2] + inputPCA[1]*p2[2] + inputPCA[2]*p3[2];
			outputTest.set_pixel(x, y, v);
		}
	}
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(outputTest.get_width(), outputTest.get_height(), false, Image::FORMAT_RGBF);
		outputTest.toImageIndexed(tmpResultRef, 0);
		tmpResultRef->save_png("testFullInverse.png");
	}
	
	//Executing a transfer for each region
	for(uint64_t i=0; i<m_nbRegions; ++i)
	{
		ImageVectorType imageWithExclusiveTransfer;
		computeExemplarWithOnlyPCAOfRegion(imageWithExclusiveTransfer, i, false);
		computeExemplarWithOnlyPCAOfRegion(imageWithExclusiveTransfer, i, true);
	}
}

void TextureSynthesizerLocallyStationary::test()
{
	int totalNumber = 0;
	int numberSuperiorTo20 = 0;
	for(int i=0; i<10; ++i)
	{
		for(int j=0; j<10; ++j)
		{
			if(i != j)
			{
				for(int k=0; k<10; ++k)
				{
					if(k != i && k != j)
					{
						if(i+j+k > 20)
						{
							++numberSuperiorTo20;
							print_line(String("Permutation (I, J, K)").replace("I", String::num_int64(i)).replace("J", String::num_int64(j)).replace("K", String::num_int64(k)));
						}
						++totalNumber;
					}
				}
			}
		}
	}
	float proba = float(numberSuperiorTo20)/totalNumber;
	print_line(String("Il y a NUM permutations possibles").replace("NUM", String::num_int64(totalNumber)));
	print_line(String("Il y a NUM permutations dont la somme est supérieure à 20").replace("NUM", String::num_int64(numberSuperiorTo20)));
	print_line(String("La probabilité d'avoir une combinaison supérieure à 20 est de NUM").replace("NUM", String::num(proba)));
	return;
}

void TextureSynthesizerLocallyStationary::_bind_methods()
{
	BIND_ENUM_CONSTANT(PCA_ONLY);
	BIND_ENUM_CONSTANT(HISTOGRAM_ONLY);
	BIND_ENUM_CONSTANT(FULL);

	ClassDB::bind_method(D_METHOD("setRegionMap", "regions"), &TextureSynthesizerLocallyStationary::setRegionMap);
	ClassDB::bind_method(D_METHOD("setMode", "mode"), &TextureSynthesizerLocallyStationary::setMode);
	ClassDB::bind_method(D_METHOD("generate"), &TextureSynthesizerLocallyStationary::generate);
	
	ClassDB::bind_method(D_METHOD("originsMapToImage", "origins"), &TextureSynthesizerLocallyStationary::originsMapToImage);
	ClassDB::bind_method(D_METHOD("simplifiedRegionMapToImage", "regionsSimplified"), &TextureSynthesizerLocallyStationary::simplifiedRegionMapToImage);
	ClassDB::bind_method(D_METHOD("invTFilteredToTexture2DArray", "invTFilteredRef", "component"), &TextureSynthesizerLocallyStationary::invTFilteredToTexture2DArray);
	ClassDB::bind_method(D_METHOD("invPCAFilteredToTexture2DArray", "invPCAFilteredRef", "component"), &TextureSynthesizerLocallyStationary::invPCAFilteredToTexture2DArray);
	
	ClassDB::bind_method(D_METHOD("regionalContributionsToTexture2DArray", "regionalContributionsRef"), &TextureSynthesizerLocallyStationary::regionalContributionsToTexture2DArray);
	ClassDB::bind_method(D_METHOD("compactContributionsToTexture2DArray", "compactContributionsRef"), &TextureSynthesizerLocallyStationary::compactContributionsToTexture2DArray);
	ClassDB::bind_method(D_METHOD("compactContributionsToImage", "compactContributionsRef"), &TextureSynthesizerLocallyStationary::compactContributionsToImage);
	ClassDB::bind_method(D_METHOD("groundTruthAtlasesTo2DArray", "atlasesRef", "originsRef", "regionRef"), &TextureSynthesizerLocallyStationary::groundTruthAtlasesTo2DArrayAlbedo);
	
	ClassDB::bind_method(D_METHOD("invTToImage", "invTRef", "component"), &TextureSynthesizerLocallyStationary::invTToImage);
	ClassDB::bind_method(D_METHOD("GaussianExemplarToImage", "GaussianExemplarRef", "component"), &TextureSynthesizerLocallyStationary::GaussianExemplarToImage);
	ClassDB::bind_method(D_METHOD("exemplarInLocalPCAsToImage", "exemplarPCARef", "component"), &TextureSynthesizerLocallyStationary::exemplarInLocalPCAsToImage);
	ClassDB::bind_method(D_METHOD("invLocalPCAToImage", "invPCARef", "component"), &TextureSynthesizerLocallyStationary::invLocalPCAToImage);
	ClassDB::bind_method(D_METHOD("setDebugSaves", "b"), &TextureSynthesizerLocallyStationary::setDebugSaves);
	ClassDB::bind_method(D_METHOD("test"), &TextureSynthesizerLocallyStationary::test);
}

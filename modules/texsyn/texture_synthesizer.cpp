#include "core/object/class_db.h"
#include "scene/resources/texture.h"
#include "texture_synthesizer.h"
#include "colorsynthesisprototype.h"
#include "contentatlas.h"

TextureSynthesizer::TextureSynthesizer() :
	m_textureTypeFlag(0),
	m_imageRefs(),
	m_exemplar(),
	m_outputImageVector()
{
	m_imageRefs.resize(9);
}

void TextureSynthesizer::set_component(TextureTypeFlag type, Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	m_textureTypeFlag = m_textureTypeFlag | type;
	m_imageRefs[texsyn_log2(type)] = image;
}

void TextureSynthesizer::outputToComponent(TextureTypeFlag type, Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(!m_outputImageVector.is_initialized(),
						"an output must be computed before calling outputToComponent.");
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & type), "corresponding component must be set first.");
	Ref<Image> tmpResultRef;
	tmpResultRef = Image::create_empty(m_outputImageVector.get_width(), m_outputImageVector.get_height(), false, Image::FORMAT_RGBF);
	unsigned int index = getStartIndexFromComponent(type);
	m_outputImageVector.toImageIndexed(tmpResultRef, index);
	image->copy_from(tmpResultRef);
}

void TextureSynthesizer::_bind_methods()
{
	BIND_ENUM_CONSTANT(ALBEDO);
	BIND_ENUM_CONSTANT(NORMAL);
	BIND_ENUM_CONSTANT(HEIGHT);
	BIND_ENUM_CONSTANT(ROUGHNESS);
	BIND_ENUM_CONSTANT(METALLIC);
	BIND_ENUM_CONSTANT(AMBIENT_OCCLUSION);
	BIND_ENUM_CONSTANT(SPECULAR);
	BIND_ENUM_CONSTANT(ALPHA);
	BIND_ENUM_CONSTANT(RIM);

	ClassDB::bind_method(D_METHOD("set_component", "component", "image"), &TextureSynthesizer::set_component);
	ClassDB::bind_method(D_METHOD("outputToComponent", "component", "image"), &TextureSynthesizer::outputToComponent);
}

SamplerTextureSynthesizer::SamplerTextureSynthesizer() : 
TextureSynthesizer(),
m_proceduralSampling(),
m_meanAccuracy(512),
m_meanSize(512)
{
	m_proceduralSampling.set_exemplar(&m_exemplar);
}

void SamplerTextureSynthesizer::set_cyclostationaryPeriods(Vector2 t0, Vector2 t1)
{
	TexSyn::SamplerPeriods *sampler = memnew(TexSyn::SamplerPeriods(0));
	sampler->setPeriods(t0, t1);
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void SamplerTextureSynthesizer::set_importancePDF(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ImageScalarType pdf;
	pdf.fromImage(image);
	TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(pdf, 0));
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void SamplerTextureSynthesizer::set_meanAccuracy(unsigned int accuracy)
{
	ERR_FAIL_COND_MSG(accuracy == 0, "accuracy must be greater than 0.");
	m_meanAccuracy = accuracy;
}

void SamplerTextureSynthesizer::set_meanSize(unsigned int meanSize)
{
	ERR_FAIL_COND_MSG(meanSize == 0, "mean size must be greater than 0.");
	m_meanSize = meanSize;
}

void SamplerTextureSynthesizer::computeWeightedMean()
{
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "normal must be set with set_normal first.");
	m_proceduralSampling.computeWeightedMean(m_outputImageVector, m_exemplar.get_width(), m_exemplar.get_height(), m_meanAccuracy);
}

void SamplerTextureSynthesizer::computeAutocovarianceSampler()
{
	ImageVectorType imagePCA;
	if(!m_exemplar.is_initialized())
		computeImageVector();
	TexSyn::PCA<float> pca(m_exemplar);
	pca.computeProjection(1);
	imagePCA.init(m_exemplar.get_width(), m_exemplar.get_height(), 1);
	pca.project(imagePCA);
	ImageScalarType imagePCScalar = imagePCA.get_image(0);
	TexSyn::StatisticsScalar<float> statistics(imagePCScalar);
	const ImageScalarType &imageAutocovariance = statistics.get_autocovariance(true);
	TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(imageAutocovariance, 0));
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void SamplerTextureSynthesizer::samplerPdfToImage(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!image->is_empty(), "image must be empty.");
	const TexSyn::SamplerImportance *si = dynamic_cast<const TexSyn::SamplerImportance *>(m_proceduralSampling.sampler());
	ERR_FAIL_COND_MSG(si == nullptr, "importance sampler must be set with computeAutocovarianceSampler().");
	Ref<Image> refPdf;
	refPdf = Image::create_empty(si->importanceFunction().get_width(), si->importanceFunction().get_height(), false, Image::FORMAT_RF);
	si->importanceFunction().toImage(refPdf, 0);
	image->copy_from(refPdf);
	return;
}

void SamplerTextureSynthesizer::samplerRealizationToImage(Ref<Image> image, unsigned int size)
{
	ERR_FAIL_COND_MSG(!image->is_empty(), "image must be empty.");
	Ref<Image> refRealization;
	refRealization = Image::create_empty(size, 1, false, Image::FORMAT_RGF);
	ImageVectorType realization;
	m_proceduralSampling.preComputeSamplerRealization(realization, size);
	realization.toImage(refRealization);
	image->copy_from(refRealization);
	return;
}

void SamplerTextureSynthesizer::centerExemplar(Ref<Image> exemplar, Ref<Image> mean)
{
	ERR_FAIL_COND_MSG(exemplar.is_null(), "exemplar must not be null.");
	ERR_FAIL_COND_MSG(exemplar->is_empty(), "exemplar must not be empty.");
	ERR_FAIL_COND_MSG(mean.is_null(), "mean must not be null.");
	ERR_FAIL_COND_MSG(mean->is_empty(), "mean must not be empty (use spatiallyVaryingMean functions).");
	Ref<Image> refMean;
	refMean = Image::create_from_data(mean->get_width(), mean->get_height(), false, mean->get_format(), mean->get_data());
	refMean->resize(exemplar->get_width(), exemplar->get_height(), Image::INTERPOLATE_CUBIC);
	ImageVectorType meanImageVector, exemplarImageVector;
	meanImageVector.fromImage(refMean);
	exemplarImageVector.fromImage(exemplar);
	ERR_FAIL_COND_MSG(meanImageVector.get_nbDimensions() != exemplarImageVector.get_nbDimensions(), "exemplar and mean must have the same number of dimensions.");
	exemplarImageVector -= meanImageVector;
	exemplarImageVector.toImage(exemplar);
	return;
}

void SamplerTextureSynthesizer::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_cyclostationaryPeriods", "t0", "t1"), &SamplerTextureSynthesizer::set_cyclostationaryPeriods);
	ClassDB::bind_method(D_METHOD("set_importancePDF", "image"), &SamplerTextureSynthesizer::set_importancePDF);
	ClassDB::bind_method(D_METHOD("set_meanAccuracy", "accuracy"), &SamplerTextureSynthesizer::set_meanAccuracy);
	ClassDB::bind_method(D_METHOD("set_meanSize", "size"), &SamplerTextureSynthesizer::set_meanSize);
	ClassDB::bind_method(D_METHOD("samplerRealizationToImage", "image", "size"), &SamplerTextureSynthesizer::samplerRealizationToImage, DEFVAL(4096));
	ClassDB::bind_method(D_METHOD("centerExemplar", "exemplar", "mean"), &SamplerTextureSynthesizer::centerExemplar);
	ClassDB::bind_method(D_METHOD("computeAutocovarianceSampler"), &SamplerTextureSynthesizer::computeAutocovarianceSampler);
	ClassDB::bind_method(D_METHOD("samplerPdfToImage", "image"), &SamplerTextureSynthesizer::samplerPdfToImage);
	ClassDB::bind_method(D_METHOD("computeWeightedMean"), &SamplerTextureSynthesizer::computeWeightedMean);
}

void TextureSynthesizer::computeImageVector()
{
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(),
						"Texture components must be set first with set_component.");
	unsigned int nbDimensions = 0;
	unsigned int width=0, height=0;
	auto addDimensions = [&nbDimensions, &width, &height, this] (int flag, int nbDimensionsExpected)
	{
		if(m_textureTypeFlag & flag)
		{
			unsigned int index = texsyn_log2(flag);
			DEV_ASSERT(!m_imageRefs[index].is_null() && m_imageRefs[index].is_valid());
			nbDimensions += nbDimensionsExpected;
			width = m_imageRefs[index]->get_width();
			height = m_imageRefs[index]->get_height();
		}
	};

	addDimensions(ALBEDO, 3);
	addDimensions(NORMAL, 3);
	addDimensions(HEIGHT, 1);
	addDimensions(ROUGHNESS, 1);
	addDimensions(METALLIC, 1);
	addDimensions(AMBIENT_OCCLUSION, 1);
	addDimensions(SPECULAR, 1);
	addDimensions(ALPHA, 1);
	addDimensions(RIM, 1);

	m_exemplar.init(width, height, nbDimensions);
	unsigned int currentIVIndex = 0;

	auto fillTexture = [&currentIVIndex, this] (int flag, int nbDimensionsExpected)
	{
		if(m_textureTypeFlag & flag)
		{
			unsigned int index = texsyn_log2(flag);
			DEV_ASSERT(!m_imageRefs[index].is_null() && m_imageRefs[index].is_valid());
			m_exemplar.fromImageIndexed(m_imageRefs[index], currentIVIndex);
			currentIVIndex += nbDimensionsExpected;
		}
	};

	fillTexture(ALBEDO, 3);
	fillTexture(NORMAL, 3);
	fillTexture(HEIGHT, 1);
	fillTexture(ROUGHNESS, 1);
	fillTexture(METALLIC, 1);
	fillTexture(AMBIENT_OCCLUSION, 1);
	fillTexture(SPECULAR, 1);
	fillTexture(ALPHA, 1);
	fillTexture(RIM, 1);

	return;
}

unsigned int TextureSynthesizer::getStartIndexFromComponent(TextureTypeFlag flag)
{
	int index = 0;
	if((m_textureTypeFlag & ALBEDO) && flag > ALBEDO)
	{
		index += 3;
	}
	if((m_textureTypeFlag & NORMAL) && flag > NORMAL)
	{
		index += 3;
	}
	if((m_textureTypeFlag & HEIGHT) && flag > HEIGHT)
	{
		index += 1;
	}
	if((m_textureTypeFlag & ROUGHNESS) && flag > ROUGHNESS)
	{
		index += 1;
	}
	if((m_textureTypeFlag & METALLIC) && flag > METALLIC)
	{
		index += 1;
	}
	return index;
}

LocallyStationaryTextureSynthesizer::LocallyStationaryTextureSynthesizer() :
TextureSynthesizer(),
m_nbRegions(0),
m_regionsInt(),
m_multiIdMap(),
m_gst(),
m_invT(),
m_TG(),
m_exemplarPCA(),
m_invPCA(),
m_invPCAPreFiltered(),
m_regionsContributions()
{}

void LocallyStationaryTextureSynthesizer::setRegionMap(Ref<Image> regions)
{
	ERR_FAIL_COND_MSG(regions.is_null(), "regions must not be null.");

	const int minSize = 512;

	//collecting all the ids
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
	
	Vector<int> count;
	count.resize(m_nbRegions);
	//Count region areas
	m_regionsInt.for_all_pixels([&] (const ImageRegionType::DataType &pix)
	{
		++count.write[pix];
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
					if(lastCuratedRegion != pix)
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
}

void LocallyStationaryTextureSynthesizer::originsMapToImage(Ref<Image> origins)
{
	ERR_FAIL_COND_MSG(origins.is_null(), "origins must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ImageVectorType imageVectorOrigins;
	imageVectorOrigins.init(m_nbRegions, 1, 2);
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
		imageVectorOrigins.set_pixel(otherID, 0, 0, dx);
		imageVectorOrigins.set_pixel(otherID, 0, 1, dy);
	}
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(imageVectorOrigins.get_width(), imageVectorOrigins.get_height(), false, Image::FORMAT_RGF);
		imageVectorOrigins.toImageIndexed(tmpResultRef, 0);
		origins->copy_from(tmpResultRef);
	}
}

void LocallyStationaryTextureSynthesizer::simplifiedRegionMapToImage(Ref<Image> regionsSimplified)
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

void LocallyStationaryTextureSynthesizer::invTFilteredToTexture2DArrayAlbedo(Ref<Texture2DArray> invTFilteredRef)
{
	ERR_FAIL_COND_MSG(invTFilteredRef.is_null(), "invTFilteredRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(!m_exemplar.is_initialized(), "computeInvT() or computeGaussianExemplar() should be called before this function.");
	//computing the filtered region map

	precomputationsPrefiltering();
	
	//Computing the pre-filtered LUTs
	LocalVector<ImageVectorType> TinvVector;
	
	Vector<Ref<Image>> invTVectorRef;
	invTVectorRef.resize(m_mipmapMultiIDMap.nbMaps());
	
	for(int i=0; i<m_mipmapMultiIDMap.nbMaps(); ++i)
	{
		ImageVectorType invT;
		invT.init(128, m_nbRegions, m_mipmapExemplar.mipmap(i).get_nbDimensions(), true);
		m_gst.computeinvTMultipleRegions(m_mipmapExemplar.mipmap(i), m_mipmapMultiIDMap.mipmap(i), invT, false);
		TinvVector.push_back(invT);
		Ref<Image> tmpResultRef = Image::create_empty(invT.get_width(), invT.get_height(), false, Image::FORMAT_RGBF);
		invT.toImageIndexed(tmpResultRef, 0);
		invTVectorRef.write[i].instantiate();
		invTVectorRef.write[i]->copy_from(tmpResultRef);
		//Optionnal previsualisation save
		if(m_debugSaves)
		{
			invTVectorRef.write[i]->save_png(String("debug/invTRef_num.png").replace("num", String::num_int64(i)));
		}
	}
	
	invTFilteredRef->create_from_images(invTVectorRef);
}

void LocallyStationaryTextureSynthesizer::invPCAFilteredToTexture2DArrayAlbedo(Ref<Texture2DArray> invPCAFilteredRef)
{
	ERR_FAIL_COND_MSG(invPCAFilteredRef.is_null(), "invTFilteredRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(m_invPCAPreFiltered.is_empty(), "computeExemplarInLocalPCAs() or computeInvLocalPCAs() should be called before this function.");
	
	Vector<Ref<Image>> invPCAVectorRef;
	invPCAVectorRef.resize(m_mipmapMultiIDMap.nbMaps());
	
	for(int i=0; i<invPCAVectorRef.size(); ++i)
	{
		const ImageVectorType &invPCA = m_invPCAPreFiltered[i];
		Ref<Image> tmpResultRef = Image::create_empty(invPCA.get_width(), invPCA.get_height(), false, Image::FORMAT_RGBF);
		invPCA.toImageIndexed(tmpResultRef, 0);
		invPCAVectorRef.write[i].instantiate();
		invPCAVectorRef.write[i]->copy_from(tmpResultRef);
		//Optionnal previsualisation save
		if(m_debugSaves)
		{
			invPCAVectorRef.write[i]->save_png(String("debug/invPCARef_num.png").replace("num", String::num_int64(i)));
		}
	}
	invPCAFilteredRef->create_from_images(invPCAVectorRef);
}

void LocallyStationaryTextureSynthesizer::invTAndPCAToTexture2DArrayAlbedo(Ref<Texture2DArray> invFilteredRef)
{
	ERR_FAIL_COND_MSG(invFilteredRef.is_null(), "invTFilteredRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	
	if(!m_exemplarPCA.is_initialized())
	{
		precomputationsLocalPCAs();
	}
	if(!m_invT.is_initialized())
	{
		precomputationsGaussian();
	}
	
	Vector<ImageVectorType> invFiltered;
	
	invFiltered.resize(m_mipmapMultiIDMap.nbMaps());
	invFiltered.write[0] = m_invT;
	
	for(int i=1; i<invFiltered.size(); ++i)
	{
		const ImageVectorType &firstInvT = invFiltered[0];
		//copy the histogram and apply a Gaussian kernel to it
		invFiltered.write[i] = firstInvT;
		ImageVectorType &currentInvT = invFiltered.write[i];
		int windowWidth = int(pow(2.0f, float(i)));
		
		for(int y=0; y<currentInvT.get_height(); ++y)
		{
			//Computing the average variance over the size of the window
			ImageVectorType::VectorType variance = footprintVariance(m_TG, i, uint64_t(y));
			for(unsigned int d=0; d<m_TG.get_nbDimensions(); ++d)
			{
				m_gst.prefilterLUT(currentInvT, variance[d], uint64_t(y), d);
			}
		}
	}
	
	Vector<Ref<Image>> invVectorRef;
	invVectorRef.resize(m_mipmapMultiIDMap.nbMaps());
	
	for(int i=0; i<invVectorRef.size(); ++i)
	{
		Ref<Image> tmpResultRef = Image::create_empty(invFiltered[i].get_width(), invFiltered[i].get_height(), false, Image::FORMAT_RGBF);
		invFiltered[i].toImageIndexed(tmpResultRef, 0);
		invVectorRef.write[i].instantiate();
		invVectorRef.write[i]->copy_from(tmpResultRef);
		//Optionnal previsualisation save
		if(m_debugSaves)
		{
			invVectorRef.write[i]->save_png(String("debug/invTFiltered_num.png").replace("num", String::num_int64(i)));
		}
	}
	invFilteredRef->create_from_images(invVectorRef);
	
	if(!m_debugSaves)
	{
		return;
	}
	
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
			
			ImageVectorType::VectorType input = m_TG.get_pixel(x, y);
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
		computeExemplarWithOnlyPCAOfRegion(imageWithExclusiveTransfer, i);
	}
	
}

void LocallyStationaryTextureSynthesizer::regionalContributionsToTexture2DArray(Ref<Texture2DArray> regionalContributionsRef)
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
			for(int k=0; k<mipmap.nbMaps(); ++k)
			{
				const ImageVectorType &map = mipmap.mipmap(k);
				Ref<Image> tmpResultRef = Image::create_empty(map.get_width(), map.get_height(), true, Image::FORMAT_RF);
				map.toImage(tmpResultRef);
				tmpResultRef->save_png(String("debug/contribution_num_mipmap_k.png").replace("num", String::num_int64(i)).replace("k", String::num_int64(k)));
			}
		}
	}
	regionalContributionsRef->create_from_images(regionalContributionsVectorRef);
}

void LocallyStationaryTextureSynthesizer::compactContributionsToTexture2DArray(Ref<Texture2DArray> compactContributionsRef)
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
		m_regionsContributions[i].upsizeMipmap();
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

void LocallyStationaryTextureSynthesizer::compactContributionsToImage(Ref<Image> compactContributionsRef)
{
	ERR_FAIL_COND_MSG(compactContributionsRef.is_null(), "regionalContributionsRef must not be null.");
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(m_regionsContributions.is_empty(), "computeExemplarInLocalPCAs() or computeInvLocalPCAs() should be called before this function.");
	
	//This version computes a mipmap instead of an array of images.
	//The first channel contains the dominant region ID and the two (three one day?) others contain the contributions.
	
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
}

void LocallyStationaryTextureSynthesizer::computeInvT()
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	if(!m_invT.is_initialized())
	{
		if(!m_exemplar.is_initialized())
			computeImageVector();
		precomputationsGaussian();
	}
	m_outputImageVector = m_invT;
}

void LocallyStationaryTextureSynthesizer::computeGaussianExemplar()
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	if(!m_TG.is_initialized())
	{
		if(!m_exemplar.is_initialized())
			computeImageVector();
		precomputationsGaussian();
	}
	m_outputImageVector = m_TG;
}

void LocallyStationaryTextureSynthesizer::computeExemplarInLocalPCAs()
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	if(!m_exemplarPCA.is_initialized())
	{
		computeImageVector();
		precomputationsLocalPCAs();
	}
	m_outputImageVector = m_exemplarPCA;
	
	//In order to save in .png, add 0.5
	m_outputImageVector.for_all_images([&] (ImageVectorType::ImageScalarType &image, unsigned int d)
	{
		image += 0.5;
	});
}

void LocallyStationaryTextureSynthesizer::computeInvLocalPCAs()
{	
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	if(!m_invPCA.is_initialized())
	{
		computeImageVector();
		precomputationsLocalPCAs();
	}
	m_outputImageVector = m_invPCA;
}

void LocallyStationaryTextureSynthesizer::groundTruthAtlasesTo2DArrayAlbedo(Ref<Texture2DArray> atlasesRef, Ref<Image> originsRef, Ref<Image> regionRef)
{
	ERR_FAIL_COND_MSG(!m_regionsInt.is_initialized(), "region map must be set with setRegionMap().");
	ERR_FAIL_COND_MSG(m_imageRefs.is_empty(), "one or more components must be set with setComponent().");
	ERR_FAIL_COND_MSG(originsRef.is_null(), "origins must not be null.");
	ERR_FAIL_COND_MSG(regionRef.is_null(), "region must not be null.");
	
	Ref<Image> tmpResultRef;
	
	//Computing atlases
	
//	ContentAtlas contentAtlas;
	
//	LocalVector<ImageVectorType> contents;
//	contents.resize(m_nbRegions);
	
//	for(unsigned int i=0; i<m_nbRegions; ++i)
//	{
//		computeExemplarWithOnlyPCAOfRegion(contents[i], i);
//	}
	
//	contentAtlas.setRegionMap(m_multiIdMap, m_nbRegions);
//	contentAtlas.setContentsFromAllVersions(contents);
//	contentAtlas.setContributions(m_regionsContributions);
	
//	contentAtlas.computeAllAtlases();
	
	//Exporting atlases
//	Vector<Ref<Image>> atlasVectorRef;
//	atlasVectorRef.resize(m_nbRegions);
	
//	for(unsigned int i=0; i<m_nbRegions; ++i)
//	{
//		const ImageVectorType &atlas = contentAtlas.atlasVector()[i];
//		Ref<Image> tmpResultRef = Image::create_empty(atlas.get_width(), atlas.get_height(), false, Image::FORMAT_RGBF);
//		atlas.toImage(tmpResultRef);
//		atlasVectorRef.write[i].instantiate();
//		atlasVectorRef.write[i]->copy_from(tmpResultRef);
//		if(m_debugSaves)
//		{
//			atlasVectorRef[i]->save_png(String("debug/atlasGroundTruth_num.png").replace("num", String::num_int64(i)));
//		}
//	}
//	atlasesRef->create_from_images(atlasVectorRef);
	
//	//Exporting origins
//	ImageVectorType origins = contentAtlas.originTexture();
//	tmpResultRef = Image::create_empty(origins.get_width(), origins.get_height(), false, Image::FORMAT_RGBF);
//	origins.toImage(tmpResultRef);
//	originsRef->copy_from(tmpResultRef);
	
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
			
			pix32 = (bit.hi >> 32) & 0xffffffff;
			ImageVectorType::DataType &pix0 = imageRegionReadable.get_pixelRef(x, y, 0);
			reinterpret_cast<uint32_t &>(pix0) = pix32;
//			print_line(String("r: bit at xx, yy: val").replace("val", String::num_uint64(pix32)).replace("xx", String::num_int64(x)).replace("yy", String::num_int64(y)));
		}
	}
	tmpResultRef = Image::create_empty(multiIDMap.get_width(), multiIDMap.get_height(), true, Image::FORMAT_RGBAF);
	tmpResultRef->mark_as_IDMap(); //custom code
	imageRegionReadable.toImage(tmpResultRef);
	tmpResultRef->generate_mipmaps(false);
	regionRef->copy_from(tmpResultRef);
}

void LocallyStationaryTextureSynthesizer::precomputationsPrefiltering()
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
			tmpResultRef->save_png(String("debug/mipmapRegions_num.png").replace("num", String::num_int64(i)));
		}
		for(int i=0; i<m_mipmapExemplar.nbMaps(); ++i)
		{
			const ImageVectorType &map = m_mipmapExemplar.mipmap(i);
			Ref<Image> tmpResultRef = Image::create_empty(map.get_width(), map.get_height(), false, Image::FORMAT_RGBF);
			map.toImageIndexed(tmpResultRef, 0);
			tmpResultRef->save_png(String("debug/mipmapExemplar_num.png").replace("num", String::num_int64(i)));
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
}

void LocallyStationaryTextureSynthesizer::precomputationsGaussian()
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
	
		m_TG.init(m_exemplarPCA.get_width(), m_exemplarPCA.get_height(), m_exemplarPCA.get_nbDimensions(), true);
		m_gst.computeTinputRegions(m_exemplarPCA, m_regionsInt, m_TG, false, false);
	}
	else
	{
		m_invT.init(invTSize, m_nbRegions, m_exemplar.get_nbDimensions(), true);
		m_gst.computeinvTRegions(m_exemplar, m_regionsInt, m_invT);
	
		m_TG.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
		m_gst.computeTinputRegions(m_exemplar, m_regionsInt, m_TG, false, false);
	}
}

void LocallyStationaryTextureSynthesizer::precomputationsLocalPCAs()
{
#define ADDGLOBALPCA
	precomputationsPrefiltering();
	m_mipmapMultiIDMap.upsizeMipmap();
	m_mipmapExemplar.upsizeMipmap();
	//computing local PCAs, projected exemplar, and packing inverse PCA infos
	m_exemplarPCA.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	LocalVector<PCAType> localPCAs;
	unsigned int regionOffset = 0;
#ifdef ADDGLOBALPCA
	regionOffset = 1;
#endif
	m_invPCA.init(1+m_exemplar.get_nbDimensions(), m_nbRegions+regionOffset, m_exemplar.get_nbDimensions(), true);
	for(unsigned int i=0; i<m_nbRegions; ++i)
	{
		localPCAs.push_back(PCAType(m_exemplar, m_multiIdMap, uint64_t(i)));
		PCAType &localPCA = localPCAs[i];
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
	m_invPCAPreFiltered.resize(m_mipmapExemplar.nbMaps());
	LocalVector<LocalVector<PCAType>> localPCAsPreFiltered;
	localPCAsPreFiltered.resize(m_mipmapExemplar.nbMaps());
	

	for(unsigned int k=0; k<localPCAsPreFiltered.size(); ++k)
	{
		LocalVector<PCAType> &localPCAs = localPCAsPreFiltered[k];
		const ImageVectorType &exemplar = m_mipmapExemplar.mipmap(k);
		const ImageMultipleRegionType &multipleRegions = m_mipmapMultiIDMap.mipmap(k);
		ImageVectorType &invPCA = m_invPCAPreFiltered[k];
		invPCA.init(1+exemplar.get_nbDimensions(), m_nbRegions, exemplar.get_nbDimensions(), true);
		for(unsigned int i=0; i<m_nbRegions; ++i)
		{
			localPCAs.push_back(PCAType(exemplar, multipleRegions, uint64_t(i), false));
			PCAType &localPCA = localPCAs[i];
			localPCA.computePCA();
			PCAType::MatrixType localEigenVectors = localPCA.get_eigenVectors().transpose();
			PCAType::VectorType localMean = localPCA.get_mean();
			//filling invPCA: at x=0, mean, and then eigen vectors
			for(unsigned int d=0; d<exemplar.get_nbDimensions(); ++d)
			{
				invPCA.set_pixel(0, i, d, localMean[d]);
				for(int r=0; r<localEigenVectors.rows(); ++r)
				{
					invPCA.set_pixel(r+1, i, d, localEigenVectors(r, d));
				}
			}
		}
	}

	//Testing by simulating the GPU process
	ImageVectorType outputPCA;
	outputPCA.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	for(int x=0; x<outputPCA.get_width(); ++x)
	{
		for(int y=0; y<outputPCA.get_height(); ++y)
		{
			int region = m_regionsInt.get_pixel(x, y);
			ImageVectorType::VectorType mean = m_invPCA.get_pixel(0, region);
			ImageVectorType::VectorType p1 = m_invPCA.get_pixel(1, region);
			ImageVectorType::VectorType p2 = m_invPCA.get_pixel(2, region);
			ImageVectorType::VectorType p3 = m_invPCA.get_pixel(3, region);
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
}

void LocallyStationaryTextureSynthesizer::computeExemplarWithOnlyPCAOfRegion(ImageVectorType &texture, uint64_t region)
{
	ERR_FAIL_COND_MSG(!m_exemplarPCA.is_initialized(), "Exemplar in PCA space must have been computed before calling this function.");
	texture.init(m_exemplar.get_width(), m_exemplar.get_height(), m_exemplar.get_nbDimensions(), true);
	for(int x=0; x<texture.get_width(); ++x)
	{
		for(int y=0; y<texture.get_height(); ++y)
		{
			ImageVectorType::VectorType inputPCA;
			
			if(m_TG.is_initialized())
			{
				ImageVectorType::VectorType input = m_TG.get_pixel(x, y);
				int r, g, b;
				r = MAX(0, MIN(input[0]*(m_invT.get_width()-1), m_invT.get_width()-1));
				g = MAX(0, MIN(input[1]*(m_invT.get_width()-1), m_invT.get_width()-1));
				b = MAX(0, MIN(input[2]*(m_invT.get_width()-1), m_invT.get_width()-1));
				inputPCA.resize(3);
				inputPCA.write[0] = m_invT.get_pixel(r, region, 0) - 0.5;
				inputPCA.write[1] = m_invT.get_pixel(g, region, 1) - 0.5;
				inputPCA.write[2] = m_invT.get_pixel(b, region, 2) - 0.5;
			}
			else
			{
				inputPCA = m_exemplarPCA.get_pixel(x, y);
			}
			ImageVectorType::VectorType mean = m_invPCA.get_pixel(0, region+1);
			ImageVectorType::VectorType p1 = m_invPCA.get_pixel(1, region+1);
			ImageVectorType::VectorType p2 = m_invPCA.get_pixel(2, region+1);
			ImageVectorType::VectorType p3 = m_invPCA.get_pixel(3, region+1);
			ImageVectorType::VectorType v = mean;
			v.write[0] += inputPCA[0]*p1[0] + inputPCA[1]*p2[0] + inputPCA[2]*p3[0];
			v.write[1] += inputPCA[0]*p1[1] + inputPCA[1]*p2[1] + inputPCA[2]*p3[1];
			v.write[2] += inputPCA[0]*p1[2] + inputPCA[1]*p2[2] + inputPCA[2]*p3[2];
			texture.set_pixel(x, y, v);
		}
	}
	if(m_debugSaves)
	{
		Ref<Image> tmpResultRef = Image::create_empty(texture.get_width(), texture.get_height(), false, Image::FORMAT_RGBF);
		texture.toImageIndexed(tmpResultRef, 0);
		tmpResultRef->save_png(String("debug/pcaFullTransfer_num.png").replace("num", String::num_int64(region)));
	}
}

LocallyStationaryTextureSynthesizer::ImageVectorType::VectorType LocallyStationaryTextureSynthesizer::footprintVariance(const ImageVectorType &texture, unsigned int level, uint64_t region)
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
	print_line(debugString);
	return averageVariance;
}

LocallyStationaryTextureSynthesizer::ImageVectorType LocallyStationaryTextureSynthesizer::debug_visualizeRegions(const ImageMultipleRegionType &map)
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

void LocallyStationaryTextureSynthesizer::setDebugSaves(bool b)
{
	m_debugSaves = b;
}

void LocallyStationaryTextureSynthesizer::test()
{

	return;
}

void LocallyStationaryTextureSynthesizer::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("setRegionMap", "regions"), &LocallyStationaryTextureSynthesizer::setRegionMap);
	ClassDB::bind_method(D_METHOD("originsMapToImage", "origins"), &LocallyStationaryTextureSynthesizer::originsMapToImage);
	ClassDB::bind_method(D_METHOD("simplifiedRegionMapToImage", "regionsSimplified"), &LocallyStationaryTextureSynthesizer::simplifiedRegionMapToImage);
	ClassDB::bind_method(D_METHOD("invTFilteredToTexture2DArrayAlbedo", "invTFilteredRef"), &LocallyStationaryTextureSynthesizer::invTFilteredToTexture2DArrayAlbedo);
	ClassDB::bind_method(D_METHOD("invPCAFilteredToTexture2DArrayAlbedo", "invPCAFilteredRef"), &LocallyStationaryTextureSynthesizer::invPCAFilteredToTexture2DArrayAlbedo);
	ClassDB::bind_method(D_METHOD("regionalContributionsToTexture2DArray", "regionalContributionsRef"), &LocallyStationaryTextureSynthesizer::regionalContributionsToTexture2DArray);
	ClassDB::bind_method(D_METHOD("compactContributionsToTexture2DArray", "compactContributionsRef"), &LocallyStationaryTextureSynthesizer::compactContributionsToTexture2DArray);
	ClassDB::bind_method(D_METHOD("compactContributionsToImage", "compactContributionsRef"), &LocallyStationaryTextureSynthesizer::compactContributionsToImage);
	ClassDB::bind_method(D_METHOD("groundTruthAtlasesTo2DArrayAlbedo", "atlasesRef", "originsRef", "regionRef"), &LocallyStationaryTextureSynthesizer::groundTruthAtlasesTo2DArrayAlbedo);
	ClassDB::bind_method(D_METHOD("invTAndPCAToTexture2DArrayAlbedo", "invFilteredRef"), &LocallyStationaryTextureSynthesizer::invTAndPCAToTexture2DArrayAlbedo);
	ClassDB::bind_method(D_METHOD("computeInvT"), &LocallyStationaryTextureSynthesizer::computeInvT);
	ClassDB::bind_method(D_METHOD("computeGaussianExemplar"), &LocallyStationaryTextureSynthesizer::computeGaussianExemplar);
	ClassDB::bind_method(D_METHOD("computeExemplarInLocalPCAs"), &LocallyStationaryTextureSynthesizer::computeExemplarInLocalPCAs);
	ClassDB::bind_method(D_METHOD("computeInvLocalPCAs"), &LocallyStationaryTextureSynthesizer::computeInvLocalPCAs);
	ClassDB::bind_method(D_METHOD("setDebugSaves", "b"), &LocallyStationaryTextureSynthesizer::setDebugSaves);
	ClassDB::bind_method(D_METHOD("test"), &LocallyStationaryTextureSynthesizer::test);
}

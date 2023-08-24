#include "core/object/class_db.h"
#include "scene/resources/texture.h"
#include "texture_synthesizer.h"
#include "colorsynthesisprototype.h"
#include "gaussian_transfer.h"
#include "pca.h"

TextureSynthesizer::TextureSynthesizer() :
	m_textureTypeFlag(0),
	m_imageRefs(),
	m_proceduralSampling(),
	m_exemplar(),
	m_weightedMean(),
	m_meanAccuracy(1024)
{
	m_imageRefs.resize(9);
	m_proceduralSampling.set_exemplar(&m_exemplar);
}

void TextureSynthesizer::set_albedo(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | ALBEDO;
	m_imageRefs[texsyn_log2(ALBEDO)] = image;
	return;
}

void TextureSynthesizer::set_normal(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | NORMAL;
	m_imageRefs[texsyn_log2(NORMAL)] = image;
	return;
}

void TextureSynthesizer::set_height(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | HEIGHT;
	m_imageRefs[texsyn_log2(HEIGHT)] = image;
	return;
}

void TextureSynthesizer::set_roughness(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | ROUGHNESS;
	m_imageRefs[texsyn_log2(ROUGHNESS)] = image;
	return;
}

void TextureSynthesizer::set_metallic(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | METALLIC;
	m_imageRefs[texsyn_log2(METALLIC)] = image;
	return;
}

void TextureSynthesizer::set_ao(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | AMBIENT_OCCLUSION;
	m_imageRefs[texsyn_log2(AMBIENT_OCCLUSION)] = image;
	return;
}

void TextureSynthesizer::set_component(TextureTypeFlag type, Ref<Image> image)
{
	switch(type)
	{
		case ALBEDO:
			set_albedo(image);
			break;
		case NORMAL:
			set_normal(image);
			break;
		case HEIGHT:
			set_height(image);
			break;
		case ROUGHNESS:
			set_roughness(image);
			break;
		case METALLIC:
			set_metallic(image);
			break;
		case AMBIENT_OCCLUSION:
			set_ao(image);
			break;
		default:
			ERR_FAIL_MSG("Component not supported yet.");
			break;
	}
}

void TextureSynthesizer::spatiallyVaryingMeanToAlbedo(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & ALBEDO),
					  "albedo must be set with set_albedo first.");
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "albedo must be set with set_albedo first.");
	if(!m_weightedMean.is_initialized())
	{
		m_proceduralSampling.weightedMean(m_weightedMean, image->get_width(), image->get_height(), m_meanAccuracy);
	}
	m_weightedMean.toImageIndexed(image, 0);
	return;
}

void TextureSynthesizer::spatiallyVaryingMeanToNormal(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & NORMAL),
					  "normal must be set with set_normal first.");
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "normal must be set with set_normal first.");
	if(!m_weightedMean.is_initialized())
	{
		m_proceduralSampling.weightedMean(m_weightedMean, image->get_width(), image->get_height(), m_meanAccuracy);
	}
	unsigned int index = 0;
	if(m_textureTypeFlag & ALBEDO)
	{
		index += 3;
	}
	m_weightedMean.toImageIndexed(image, index);
	return;
}

void TextureSynthesizer::spatiallyVaryingMeanToHeight(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & HEIGHT),
					  "height must be set with set_height first.");
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "height must be set with set_height first.");
	if(!m_weightedMean.is_initialized())
	{
		m_proceduralSampling.weightedMean(m_weightedMean, image->get_width(), image->get_height(), m_meanAccuracy);
	}
	unsigned int index = 0;
	if(m_textureTypeFlag & ALBEDO)
	{
		index += 3;
	}
	if(m_textureTypeFlag & NORMAL)
	{
		index += 3;
	}
	m_weightedMean.toImageIndexed(image, index);
	return;
}

void TextureSynthesizer::spatiallyVaryingMeanToRoughness(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & ROUGHNESS),
					  "roughness must be set with set_roughness first.");
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "roughness must be set with set_roughness first.");
	if(!m_weightedMean.is_initialized())
	{
		m_proceduralSampling.weightedMean(m_weightedMean, image->get_width(), image->get_height(), m_meanAccuracy);
	}
	unsigned int index = 0;
	if(m_textureTypeFlag & ALBEDO)
	{
		index += 3;
	}
	if(m_textureTypeFlag & NORMAL)
	{
		index += 3;
	}
	if(m_textureTypeFlag & HEIGHT)
	{
		index += 1;
	}
	m_weightedMean.toImageIndexed(image, index);
	return;
}

void TextureSynthesizer::spatiallyVaryingMeanToMetallic(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & METALLIC),
					  "metallic must be set with set_metallic first.");
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "metallic must be set with set_metallic first.");
	if(!m_weightedMean.is_initialized())
	{
		m_proceduralSampling.weightedMean(m_weightedMean, image->get_width(), image->get_height(), m_meanAccuracy);
	}
	unsigned int index = 0;
	if(m_textureTypeFlag & ALBEDO)
	{
		index += 3;
	}
	if(m_textureTypeFlag & NORMAL)
	{
		index += 3;
	}
	if(m_textureTypeFlag & HEIGHT)
	{
		index += 1;
	}
	if(m_textureTypeFlag & ROUGHNESS)
	{
		index += 1;
	}
	m_weightedMean.toImageIndexed(image, index);
	return;
}

void TextureSynthesizer::spatiallyVaryingMeanToAO(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!(m_textureTypeFlag & AMBIENT_OCCLUSION),
					  "ambient occlusion must be set with set_ao first.");
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "ambient occlusion must be set with set_ao first.");
	if(!m_weightedMean.is_initialized())
	{
		m_proceduralSampling.weightedMean(m_weightedMean, image->get_width(), image->get_height(), m_meanAccuracy);
	}
	unsigned int index = 0;
	if(m_textureTypeFlag & ALBEDO)
	{
		index += 3;
	}
	if(m_textureTypeFlag & NORMAL)
	{
		index += 3;
	}
	if(m_textureTypeFlag & HEIGHT)
	{
		index += 1;
	}
	if(m_textureTypeFlag & ROUGHNESS)
	{
		index += 1;
	}
	if(m_textureTypeFlag & METALLIC)
	{
		index += 1;
	}
	m_weightedMean.toImageIndexed(image, index);
	return;
}

void TextureSynthesizer::spatiallyVaryingMeanToComponent(TextureTypeFlag type, Ref<Image> image)
{
	switch(type)
	{
		case ALBEDO:
			spatiallyVaryingMeanToAlbedo(image);
			break;
		case NORMAL:
			spatiallyVaryingMeanToNormal(image);
			break;
		case HEIGHT:
			spatiallyVaryingMeanToHeight(image);
			break;
		case ROUGHNESS:
			spatiallyVaryingMeanToRoughness(image);
			break;
		case METALLIC:
			spatiallyVaryingMeanToMetallic(image);
			break;
		case AMBIENT_OCCLUSION:
			spatiallyVaryingMeanToAO(image);
			break;
		default:
			ERR_FAIL_MSG("Component not supported yet.");
			break;
	}
}

void TextureSynthesizer::set_cyclostationaryPeriods(Vector2 t0, Vector2 t1)
{
	TexSyn::SamplerPeriods *sampler = memnew(TexSyn::SamplerPeriods(0));
	sampler->setPeriods(t0, t1);
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void TextureSynthesizer::set_importancePDF(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	TexSyn::ImageScalar<float> pdf;
	pdf.fromImage(image);
	TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(pdf, 0));
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void TextureSynthesizer::set_meanAccuracy(unsigned int accuracy)
{
	m_meanAccuracy = accuracy;
}

void TextureSynthesizer::set_meanSize(unsigned int meanSize)
{
	m_meanSize = meanSize;
}

void TextureSynthesizer::computeAutocovarianceSampler()
{
	TexSyn::ImageVector<float> imagePCA;
	if(!m_exemplar.is_initialized())
		computeImageVector();
	TexSyn::PCA<float> pca(m_exemplar);
	pca.computePCA(1);
	imagePCA.init(m_exemplar.get_width(), m_exemplar.get_height(), 1);
	pca.project(imagePCA);
	TexSyn::ImageScalar<float> imagePCScalar = imagePCA.get_image(0);
	TexSyn::StatisticsScalar<float> statistics(imagePCScalar);
	const TexSyn::ImageScalar<float> &imageAutocovariance = statistics.get_autocovariance(true);
	TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(imageAutocovariance, 0));
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void TextureSynthesizer::samplerPdfToImage(Ref<Image> image)
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

void TextureSynthesizer::samplerRealizationToImage(Ref<Image> image, unsigned int size)
{
	ERR_FAIL_COND_MSG(!image->is_empty(), "image must be empty.");
	Ref<Image> refRealization;
	refRealization = Image::create_empty(size, 1, false, Image::FORMAT_RGF);
	TexSyn::ImageVector<float> realization;
	m_proceduralSampling.preComputeSamplerRealization(realization, size);
	realization.toImage(refRealization);
	image->copy_from(refRealization);
	return;
}

void TextureSynthesizer::centerExemplar(Ref<Image> exemplar, Ref<Image> mean)
{
	ERR_FAIL_COND_MSG(exemplar.is_null(), "exemplar must not be null.");
	ERR_FAIL_COND_MSG(exemplar->is_empty(), "exemplar must not be empty.");
	ERR_FAIL_COND_MSG(mean.is_null(), "mean must not be null.");
	ERR_FAIL_COND_MSG(mean->is_empty(), "mean must not be empty (use spatiallyVaryingMean functions).");
	Ref<Image> refMean;
	refMean = Image::create_from_data(mean->get_width(), mean->get_height(), false, mean->get_format(), mean->get_data());
	refMean->resize(exemplar->get_width(), exemplar->get_height(), Image::INTERPOLATE_CUBIC);
	TexSyn::ImageVector<float> meanImageVector, exemplarImageVector;
	meanImageVector.fromImage(refMean);
	exemplarImageVector.fromImage(exemplar);
	ERR_FAIL_COND_MSG(meanImageVector.get_nbDimensions() != exemplarImageVector.get_nbDimensions(), "exemplar and mean must have the same number of dimensions.");
	exemplarImageVector -= meanImageVector;
	exemplarImageVector.toImage(exemplar);
	return;
}

void TextureSynthesizer::test_colorSynthesisPrototype(Ref<Image> exemplar, Ref<Image> regions, Ref<Image> fgbgmap, Ref<Image> resultRef, Ref<Image> debugDataRef)
{
	ERR_FAIL_COND_MSG(exemplar.is_null(), "exemplar must not be null.");
	ERR_FAIL_COND_MSG(exemplar->is_empty(), "exemplar must not be empty.");
	ERR_FAIL_COND_MSG(regions.is_null(), "regions must not be null.");
	ERR_FAIL_COND_MSG(regions->is_empty(), "regions must not be empty.");
	ERR_FAIL_COND_MSG(fgbgmap.is_null(), "fgbgmap must not be null.");
	ERR_FAIL_COND_MSG(fgbgmap->is_empty(), "fgbgmap must not be empty.");
	ERR_FAIL_COND_MSG(resultRef.is_null(), "debugResult must not be null.");
	ERR_FAIL_COND_MSG(!resultRef->is_empty(), "fgbgmap must be empty.");
	
	//creating the id map with integers from regions
	using ImageRegionType = TexSyn::ImageScalar<int>;
	using MapType = HashMap<Color, int>;
	ImageRegionType regionsInt;
	MapType histogramRegions;
	regionsInt.init(exemplar->get_width(), exemplar->get_height(), true);
	int id = 1;
	regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
	{
		Color c = regions->get_pixel(x, y);
		if(c.is_equal_approx(Color(1, 1, 1)))
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
				MapType::Iterator it2 = histogramRegions.insert(c, id);
				++id;
				pix = it2->value;
			}
		}
	});
	
	TexSyn::ColorSynthesisPrototype csp;
	TexSyn::GaussianTransfer gst;

	TexSyn::ImageVector<double> exemplarIV;
	exemplarIV.fromImage(exemplar);
	csp.setExemplar(exemplarIV);
	
	TexSyn::ImageVector<double> Tinv;
	TexSyn::ImageVector<double> TG;
	Tinv.init(128, id, exemplarIV.get_nbDimensions(), true);
	TG.init(exemplarIV.get_width(), exemplarIV.get_height(), exemplarIV.get_nbDimensions(), true);
	gst.computeTinputRegions(exemplarIV, regionsInt, TG, false, false);
	gst.computeinvTRegions(exemplarIV, regionsInt, Tinv, false);
	//TexSyn::GaussianTransfer::invTMultipleRegions(exemplarIV, Tinv, regionsInt);
	
	RandomNumberGenerator rng;
	
	Ref<Image> tmpResultRef;
	tmpResultRef = Image::create_empty(exemplarIV.get_width(), exemplarIV.get_height(), false, Image::FORMAT_RGBF);
	TexSyn::ImageVector<double> result;
	result.init(exemplarIV.get_width(), exemplarIV.get_height(), exemplarIV.get_nbDimensions(), true);
	result.get_image(0).for_all_pixels([&] (TexSyn::ImageVector<double>::DataType &pix, int x, int y)
	{
		int region = regionsInt.get_pixel(x, y);
		if(region != 0)
		{
			TexSyn::ImageVector<double>::VectorType p = TG.get_pixel(x, y);
			rng.set_seed(region);
			int regionSubstitute = rng.randi_range(1, id-1);
			gst.invTRegions(p, Tinv, regionSubstitute);
			result.set_pixel(x, y, p);
		}
		else
		{
			TexSyn::ImageVector<double>::VectorType p = TG.get_pixel(x, y);
			gst.invTRegions(p, Tinv, 0);
			result.set_pixel(x, y, p);
		}
	});

//	result.get_image(0).for_all_pixels([&] (TexSyn::ImageVector<double>::DataType &pix, int x, int y)
//	{
//		int region = regionsInt.get_pixel(x, y);
//		if(region != 0 || region == 0)
//		{
//			TexSyn::ImageVector<double>::VectorType p = TG.get_pixel(x, y);
//			gst.invTMultipleRegions(p, Tinv, region);
//			result.set_pixel(x, y, p);
//		}
//	});
	
	result.toImageIndexed(tmpResultRef, 0);
	resultRef->copy_from(tmpResultRef);
	
	if(!debugDataRef.is_null())
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(TG.get_width(), TG.get_height(), false, Image::FORMAT_RGBF);
		TG.toImageIndexed(tmpResultRef, 0);
		debugDataRef->copy_from(tmpResultRef);
	}
	return;
}

void TextureSynthesizer::precomputeLocallyStationary(	Ref<Image> exemplar, Ref<Image> regions, Ref<Image> gaussianOutputRef, 
														Ref<Image> invTRef, Ref<Image> regionsOutputRef, Ref<Image> originsRef, 
														Ref<Texture2DArray> invTFilteredRef)
{
	ERR_FAIL_COND_MSG(exemplar.is_null(), "exemplar must not be null.");
	ERR_FAIL_COND_MSG(exemplar->is_empty(), "exemplar must not be empty.");
	ERR_FAIL_COND_MSG(regions.is_null(), "regions must not be null.");
	ERR_FAIL_COND_MSG(regions->is_empty(), "regions must not be empty.");
	ERR_FAIL_COND_MSG(gaussianOutputRef.is_null(), "debugResult must not be null.");
	ERR_FAIL_COND_MSG(!gaussianOutputRef->is_empty(), "fgbgmap must be empty.");
	
	//creating the id map with integers from regions
	using ImageRegionType = TexSyn::ImageScalar<int>;
	using MapType = HashMap<Color, int>;
	ImageRegionType regionsInt;
	MapType histogramRegions;
	regionsInt.init(exemplar->get_width(), exemplar->get_height(), true);
	int id = 1;
	regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
	{
		Color c = regions->get_pixel(x, y);
		if(c.is_equal_approx(Color(1, 1, 1)))
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
				MapType::Iterator it2 = histogramRegions.insert(c, id);
				++id;
				pix = it2->value;
			}
		}
	});
	
	
	//Constructing the origins of each region for seeding in the shader
	TexSyn::ImageVector<float> imageOrigins;
	imageOrigins.init(id, 1, 2);
	for(int otherID=0; otherID<id; ++otherID)
	{
		float dx = 0.0, dy = 0.0;
		int maxX=0, maxY=0, minX=regionsInt.get_width()-1, minY=regionsInt.get_height()-1;
		regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
		{
			if(pix == otherID)
			{
				maxX = std::max(maxX, x);
				maxY = std::max(maxY, y);
				minX = std::min(minX, x);
				minY = std::min(minY, y);
			}
		});
		if(maxX == regionsInt.get_width()-1 && minX == 0)
		{
			dx = 0.5;
		}
		if(maxY == regionsInt.get_width()-1 && minY == 0)
		{
			dy = 0.5;
		}
		imageOrigins.set_pixel(otherID, 0, 0, dx);
		imageOrigins.set_pixel(otherID, 0, 1, dy);
	}
	
	
	TexSyn::ColorSynthesisPrototype csp;
	TexSyn::GaussianTransfer gst;

	TexSyn::ImageVector<double> exemplarIV;
	exemplarIV.fromImage(exemplar);
	csp.setExemplar(exemplarIV);
	
	TexSyn::ImageVector<double> TG;
	TG.init(exemplarIV.get_width(), exemplarIV.get_height(), exemplarIV.get_nbDimensions(), true);
	gst.computeTinputRegions(exemplarIV, regionsInt, TG, false, false);
	
	//computing the filtered LUT
	TexSyn::MipmapMultiIDMap mipmapMultiIDMap;
	TexSyn::MipmapMultiIDMap::ImageMultiIDMapType multiIdMap;
	gst.toMultipleRegions(multiIdMap, regionsInt);
	mipmapMultiIDMap.setIDMap(multiIdMap);
	mipmapMultiIDMap.computeMipmap();
	mipmapMultiIDMap.upsizeMipmap();
	
	TexSyn::Mipmap mipmapExemplar;
	mipmapExemplar.setTexture(exemplarIV);
	mipmapExemplar.computeMipmap();
	mipmapExemplar.upsizeMipmap();
	
	TexSyn::ImageVector<double> Tinv;
	Tinv.init(128, id, TG.get_nbDimensions(), true);
	gst.computeinvTRegions(exemplarIV, regionsInt, Tinv);
	
	//Computing the pre-filtered LUTs
	LocalVector<TexSyn::ImageVector<double>> TinvVector;
	
	Vector<Ref<Image>> TinvVectorRef;
	TinvVectorRef.resize(mipmapMultiIDMap.nbMaps());
	
	for(int i=0; i<mipmapMultiIDMap.nbMaps(); ++i)
	{
		TexSyn::ImageVector<double> Tinv;
		Tinv.init(128, id, mipmapExemplar.mipmap(i).get_nbDimensions(), true);
		gst.computeinvTMultipleRegions(mipmapExemplar.mipmap(i), mipmapMultiIDMap.mipmap(i), Tinv, false);
		TinvVector.push_back(Tinv);
		Ref<Image> tmpResultRef = Image::create_empty(Tinv.get_width(), Tinv.get_height(), false, Image::FORMAT_RGBF);
		Tinv.toImageIndexed(tmpResultRef, 0);
		TinvVectorRef.write[i].instantiate();
		TinvVectorRef.write[i]->copy_from(tmpResultRef);
		TinvVectorRef.write[i]->save_png(String("invTRef_num.png").replace("num", String::num_int64(i)));
	}
	
	invTFilteredRef->create_from_images(TinvVectorRef);
	
	TexSyn::ImageScalar<double> regionsOutput;
	regionsOutput.init(regions->get_width(), regions->get_height(), true);
	regionsOutput.for_all_pixels([&] (double & pix, int x, int y)
	{
		int region = regionsInt.get_pixel(x, y);
		pix = float(region)/(id-1);
	});
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(Tinv.get_width(), Tinv.get_height(), false, Image::FORMAT_RGBF);
		Tinv.toImageIndexed(tmpResultRef, 0);
		invTRef->copy_from(tmpResultRef);
	}
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(TG.get_width(), TG.get_height(), false, Image::FORMAT_RGBF);
		TG.toImageIndexed(tmpResultRef, 0);
		gaussianOutputRef->copy_from(tmpResultRef);
	}
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(regionsOutput.get_width(), regionsOutput.get_height(), false, Image::FORMAT_RF);
		regionsOutput.toImage(tmpResultRef, 0);
		regionsOutputRef->copy_from(tmpResultRef);
	}
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(imageOrigins.get_width(), imageOrigins.get_height(), false, Image::FORMAT_RGF);
		imageOrigins.toImageIndexed(tmpResultRef, 0);
		originsRef->copy_from(tmpResultRef);
	}
	
	return;
}

void TextureSynthesizer::precomputeLocalPCA(Ref<Image> exemplar, Ref<Image> regions, Ref<Image> pcaOutputRef, Ref<Image> invPCARef, 
											Ref<Image> regionsOutputRef, Ref<Image> originsRef)
{
	ERR_FAIL_COND_MSG(exemplar.is_null(), "exemplar must not be null.");
	ERR_FAIL_COND_MSG(exemplar->is_empty(), "exemplar must not be empty.");
	ERR_FAIL_COND_MSG(regions.is_null(), "regions must not be null.");
	ERR_FAIL_COND_MSG(regions->is_empty(), "regions must not be empty.");
	ERR_FAIL_COND_MSG(pcaOutputRef.is_null(), "debugResult must not be null.");
	ERR_FAIL_COND_MSG(!pcaOutputRef->is_empty(), "fgbgmap must be empty.");
	
	//creating the id map with integers from regions
	using ImageRegionType = TexSyn::ImageScalar<int>;
	using MapType = HashMap<Color, int>;
	ImageRegionType regionsInt;
	MapType histogramRegions;
	regionsInt.init(exemplar->get_width(), exemplar->get_height(), true);
	int id = 1;
	regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
	{
		Color c = regions->get_pixel(x, y);
		if(c.is_equal_approx(Color(1, 1, 1)))
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
				MapType::Iterator it2 = histogramRegions.insert(c, id);
				++id;
				pix = it2->value;
			}
		}
	});
	
	
	//Constructing the origins of each region for seeding in the shader
	TexSyn::ImageVector<float> imageOrigins;
	imageOrigins.init(id, 1, 2);
	for(int otherID=0; otherID<id; ++otherID)
	{
		float dx = 0.0, dy = 0.0;
		int maxX=0, maxY=0, minX=regionsInt.get_width()-1, minY=regionsInt.get_height()-1;
		regionsInt.for_all_pixels([&] (ImageRegionType::DataType &pix, int x, int y)
		{
			if(pix == otherID)
			{
				maxX = std::max(maxX, x);
				maxY = std::max(maxY, y);
				minX = std::min(minX, x);
				minY = std::min(minY, y);
			}
		});
		if(maxX == regionsInt.get_width()-1 && minX == 0)
		{
			dx = 0.5;
		}
		if(maxY == regionsInt.get_width()-1 && minY == 0)
		{
			dy = 0.5;
		}
		imageOrigins.set_pixel(otherID, 0, 0, dx);
		imageOrigins.set_pixel(otherID, 0, 1, dy);
	}
	
	
	TexSyn::ColorSynthesisPrototype csp;
	TexSyn::GaussianTransfer gst;

	TexSyn::ImageVector<double> exemplarIV;
	exemplarIV.fromImage(exemplar);
	csp.setExemplar(exemplarIV);
	
	//computing the pre-filtered LUT
	TexSyn::MipmapMultiIDMap mipmapMultiIDMap;
	TexSyn::MipmapMultiIDMap::ImageMultiIDMapType multiIdMap;
	gst.toMultipleRegions(multiIdMap, regionsInt);
	mipmapMultiIDMap.setIDMap(multiIdMap);
	mipmapMultiIDMap.computeMipmap();
	mipmapMultiIDMap.upsizeMipmap();
	
	//generating the mipmap of the exemplar
	TexSyn::Mipmap mipmapExemplar;
	mipmapExemplar.setTexture(exemplarIV);
	mipmapExemplar.computeMipmap();
	mipmapExemplar.upsizeMipmap();
	
	//computing local PCAs, projected exemplar, and packing inverse PCA infos
	TexSyn::ImageVector<double> PCAOutput;
	PCAOutput.init(exemplarIV.get_width(), exemplarIV.get_height(), exemplarIV.get_nbDimensions(), true);
	using PCAType = TexSyn::PCA<double>;
	LocalVector<PCAType> localPCAs;
	TexSyn::ImageVector<double> invPCA;
	invPCA.init(1+exemplarIV.get_nbDimensions(), id, exemplarIV.get_nbDimensions(), true);
	for(int i=0; i<id; ++i)
	{
		localPCAs.push_back(PCAType(exemplarIV, multiIdMap, uint64_t(i)));
		localPCAs[i].computePCA();
		localPCAs[i].project(PCAOutput);
		PCAType::MatrixType localEigenVectors = localPCAs[i].get_eigenVectors();
		PCAType::VectorType localMean = localPCAs[i].get_mean();
		//filling invPCA: at x=0, mean, and then eigen vectors
		for(unsigned int d=0; d<exemplarIV.get_nbDimensions(); ++d)
		{
			invPCA.set_pixel(0, i, d, localMean[d]);
			for(int r=0; r<localEigenVectors.rows(); ++r)
			{
				invPCA.set_pixel(r+1, i, d, localEigenVectors(r, d));
			}
		}
	}
	//In order to save in .png, add 0.5
	PCAOutput.for_all_images([&] (TexSyn::ImageVector<double>::ImageScalarType &image, unsigned int d)
	{
		image.for_all_pixels([&] (TexSyn::ImageVector<double>::ImageScalarType::DataType &pix)
		{
			pix += 0.5;
		});
	});
	
	Vector<Ref<Image>> TinvVectorRef;
	TinvVectorRef.resize(mipmapMultiIDMap.nbMaps());
	
//	for(int i=0; i<mipmapMultiIDMap.nbMaps(); ++i)
//	{
//		TexSyn::ImageVector<double> Tinv;
//		Tinv.init(128, id, mipmapExemplar.mipmap(i).get_nbDimensions(), true);
//		gst.computeinvTMultipleRegions(mipmapExemplar.mipmap(i), mipmapMultiIDMap.mipmap(i), Tinv, false);
//		TinvVector.push_back(Tinv);
//		Ref<Image> tmpResultRef = Image::create_empty(Tinv.get_width(), Tinv.get_height(), false, Image::FORMAT_RGBF);
//		Tinv.toImageIndexed(tmpResultRef, 0);
//		TinvVectorRef.write[i].instantiate();
//		TinvVectorRef.write[i]->copy_from(tmpResultRef);
//		TinvVectorRef.write[i]->save_png(String("invTRef_num.png").replace("num", String::num_int64(i)));
//	}
	
	TexSyn::ImageScalar<double> regionsOutput;
	regionsOutput.init(regions->get_width(), regions->get_height(), true);
	regionsOutput.for_all_pixels([&] (double & pix, int x, int y)
	{
		int region = regionsInt.get_pixel(x, y);
		pix = float(region)/(id-1);
	});
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(invPCA.get_width(), invPCA.get_height(), false, Image::FORMAT_RGBF);
		invPCA.toImageIndexed(tmpResultRef, 0);
		invPCARef->copy_from(tmpResultRef);
	}
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(PCAOutput.get_width(), PCAOutput.get_height(), false, Image::FORMAT_RGBF);
		PCAOutput.toImageIndexed(tmpResultRef, 0);
		pcaOutputRef->copy_from(tmpResultRef);
	}
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(regionsOutput.get_width(), regionsOutput.get_height(), false, Image::FORMAT_RF);
		regionsOutput.toImage(tmpResultRef, 0);
		regionsOutputRef->copy_from(tmpResultRef);
	}
	
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(imageOrigins.get_width(), imageOrigins.get_height(), false, Image::FORMAT_RGF);
		imageOrigins.toImageIndexed(tmpResultRef, 0);
		originsRef->copy_from(tmpResultRef);
	}
	
	return;
}

void TextureSynthesizer::computeImageVector()
{
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

	ClassDB::bind_method(D_METHOD("spatiallyVaryingMeanToComponent", "component", "image"), &TextureSynthesizer::spatiallyVaryingMeanToComponent);

	ClassDB::bind_method(D_METHOD("set_cyclostationaryPeriods", "t0", "t1"), &TextureSynthesizer::set_cyclostationaryPeriods);
	ClassDB::bind_method(D_METHOD("set_importancePDF", "image"), &TextureSynthesizer::set_importancePDF);
	ClassDB::bind_method(D_METHOD("set_meanAccuracy", "accuracy"), &TextureSynthesizer::set_meanAccuracy);
	ClassDB::bind_method(D_METHOD("set_meanSize", "size"), &TextureSynthesizer::set_meanSize);
	ClassDB::bind_method(D_METHOD("samplerRealizationToImage", "image", "size"), &TextureSynthesizer::samplerRealizationToImage, DEFVAL(4096));
	ClassDB::bind_method(D_METHOD("centerExemplar", "exemplar", "mean"), &TextureSynthesizer::centerExemplar);
	ClassDB::bind_method(D_METHOD("computeAutocovarianceSampler"), &TextureSynthesizer::computeAutocovarianceSampler);
	ClassDB::bind_method(D_METHOD("samplerPdfToImage", "image"), &TextureSynthesizer::samplerPdfToImage);
	
	ClassDB::bind_method(D_METHOD("test_colorSynthesisPrototype", "exemplar", "regions", "fgbgmap", "result", "debug"), &TextureSynthesizer::test_colorSynthesisPrototype);

	ClassDB::bind_method(D_METHOD("precomputeLocallyStationary", "exemplar", "regions", "gaussianOutput", "invT", "regionsOutput", "originsRef", "invTFiltered"), &TextureSynthesizer::precomputeLocallyStationary);
	ClassDB::bind_method(D_METHOD("precomputeLocalPCA", "exemplar", "regions", "PCAOutput", "invPCA", "regionsOutput", "originsRef"), &TextureSynthesizer::precomputeLocalPCA);
}

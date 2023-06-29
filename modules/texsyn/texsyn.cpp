#include "core/object/class_db.h"
#include "image_pyramid.h"
#include "texsyn.h"
#include "colorsynthesisprototype.h"

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

#ifdef TEXSYN_TESTS

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

//    TexSyn::RieszPyramid<double> pyr;
//    pyr.phase_congruency(0, 4);

	return b;
}

ProceduralSampling::ProceduralSampling() :
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

void ProceduralSampling::set_albedo(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | ALBEDO;
	m_imageRefs[texsyn_log2(ALBEDO)] = image;
	return;
}

void ProceduralSampling::set_normal(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | NORMAL;
	m_imageRefs[texsyn_log2(NORMAL)] = image;
	return;
}

void ProceduralSampling::set_height(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | HEIGHT;
	m_imageRefs[texsyn_log2(HEIGHT)] = image;
	return;
}

void ProceduralSampling::set_roughness(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | ROUGHNESS;
	m_imageRefs[texsyn_log2(ROUGHNESS)] = image;
	return;
}

void ProceduralSampling::set_metallic(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | METALLIC;
	m_imageRefs[texsyn_log2(METALLIC)] = image;
	return;
}

void ProceduralSampling::set_ao(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ERR_FAIL_COND_MSG(m_exemplar.is_initialized(),
					  "you must set every map before attempting to get any spatially varying mean.");
	m_textureTypeFlag = m_textureTypeFlag | AMBIENT_OCCLUSION;
	m_imageRefs[texsyn_log2(AMBIENT_OCCLUSION)] = image;
	return;
}

void ProceduralSampling::set_component(TextureTypeFlag type, Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToAlbedo(Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToNormal(Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToHeight(Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToRoughness(Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToMetallic(Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToAO(Ref<Image> image)
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

void ProceduralSampling::spatiallyVaryingMeanToComponent(TextureTypeFlag type, Ref<Image> image)
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

void ProceduralSampling::set_cyclostationaryPeriods(Vector2 t0, Vector2 t1)
{
	TexSyn::SamplerPeriods *sampler = memnew(TexSyn::SamplerPeriods(0));
	sampler->setPeriods(t0, t1);
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void ProceduralSampling::set_importancePDF(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	TexSyn::ImageScalar<float> pdf;
	pdf.fromImage(image);
	TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(pdf, 0));
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void ProceduralSampling::set_meanAccuracy(unsigned int accuracy)
{
	m_meanAccuracy = accuracy;
}

void ProceduralSampling::set_meanSize(unsigned int meanSize)
{
	m_meanSize = meanSize;
}

void ProceduralSampling::computeAutocovarianceSampler()
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

void ProceduralSampling::samplerPdfToImage(Ref<Image> image)
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

void ProceduralSampling::samplerRealizationToImage(Ref<Image> image, unsigned int size)
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

void ProceduralSampling::centerExemplar(Ref<Image> exemplar, Ref<Image> mean)
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

void ProceduralSampling::test_colorSynthesisPrototype(Ref<Image> exemplar, Ref<Image> regions, Ref<Image> fgbgmap, Ref<Image> resultRef, Ref<Image> debugDataRef)
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
	regionsInt.init(exemplar->get_width(), exemplar->get_height());
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
	
	TexSyn::ImageVector<double> exemplarIV;
	exemplarIV.fromImage(exemplar);
	csp.setExemplar(exemplarIV);
	
	TexSyn::ImageScalar<double> fgbgmapIV;
	fgbgmapIV.fromImage(fgbgmap);
	csp.setFGBGMap(fgbgmapIV);
	
	csp.setForegroundRegionMap(regionsInt);
	
	csp.computeMeans();
	csp.computeForegroundCovarianceMatrix();
	csp.initNormalMultivariateDistribution();
	
	Ref<Image> tmpResultRef;
	tmpResultRef = Image::create_empty(exemplar->get_width(), exemplar->get_height(), false, Image::FORMAT_RGF);
	TexSyn::ImageVector<double> result = csp.createFullTexture();
	result.toImageIndexed(tmpResultRef, 0);
	resultRef->copy_from(tmpResultRef);
	
	if(!debugDataRef.is_null())
	{
		Ref<Image> tmpResultRef;
		tmpResultRef = Image::create_empty(exemplar->get_width(), exemplar->get_height(), false, Image::FORMAT_RGF);
		TexSyn::ImageVector<double> localMeans = csp.getLocalMeans();
		localMeans.toImageIndexed(tmpResultRef, 0);
		debugDataRef->copy_from(tmpResultRef);
	}
	return;
}

void ProceduralSampling::_bind_methods()
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

	ClassDB::bind_method(D_METHOD("set_component", "component", "image"), &ProceduralSampling::set_component);

	ClassDB::bind_method(D_METHOD("spatiallyVaryingMeanToComponent", "component", "image"), &ProceduralSampling::spatiallyVaryingMeanToComponent);

	ClassDB::bind_method(D_METHOD("set_cyclostationaryPeriods", "t0", "t1"), &ProceduralSampling::set_cyclostationaryPeriods);
	ClassDB::bind_method(D_METHOD("set_importancePDF", "image"), &ProceduralSampling::set_importancePDF);
	ClassDB::bind_method(D_METHOD("set_meanAccuracy", "accuracy"), &ProceduralSampling::set_meanAccuracy);
	ClassDB::bind_method(D_METHOD("set_meanSize", "size"), &ProceduralSampling::set_meanSize);
	ClassDB::bind_method(D_METHOD("samplerRealizationToImage", "image", "size"), &ProceduralSampling::samplerRealizationToImage, DEFVAL(4096));
	ClassDB::bind_method(D_METHOD("centerExemplar", "exemplar", "mean"), &ProceduralSampling::centerExemplar);
	ClassDB::bind_method(D_METHOD("computeAutocovarianceSampler"), &ProceduralSampling::computeAutocovarianceSampler);
	ClassDB::bind_method(D_METHOD("samplerPdfToImage", "image"), &ProceduralSampling::samplerPdfToImage);
	
	ClassDB::bind_method(D_METHOD("test_colorSynthesisPrototype", "exemplar", "regions", "fgbgmap", "result"), &ProceduralSampling::test_colorSynthesisPrototype);
}

void ProceduralSampling::computeImageVector()
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

#endif //ifdef TEXSYN_TESTS

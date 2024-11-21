#include "texture_synthesizer_sampler.h"

TextureSynthesizerSampler::TextureSynthesizerSampler() : 
TextureSynthesizer(),
m_proceduralSampling(),
m_meanAccuracy(512),
m_meanSize(512),
m_mean()
{
	m_proceduralSampling.set_exemplar(&m_exemplar);
}

void TextureSynthesizerSampler::setCyclostationaryPeriods(Vector2 t0, Vector2 t1)
{
	TexSyn::SamplerPeriods *sampler = memnew(TexSyn::SamplerPeriods(0));
	sampler->setPeriods(t0, t1);
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void TextureSynthesizerSampler::setImportancePDF(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(image->is_empty(), "image must not be empty.");
	ImageScalarType pdf;
	pdf.fromImage(image);
	TexSyn::SamplerImportance *sampler = memnew(TexSyn::SamplerImportance(pdf, 0));
	m_proceduralSampling.set_sampler(sampler);
	return;
}

void TextureSynthesizerSampler::setMeanAccuracy(unsigned int accuracy)
{
	ERR_FAIL_COND_MSG(accuracy == 0, "accuracy must be greater than 0.");
	m_meanAccuracy = accuracy;
}

void TextureSynthesizerSampler::setMeanSize(unsigned int meanSize)
{
	ERR_FAIL_COND_MSG(meanSize == 0, "mean size must be greater than 0.");
	m_meanSize = meanSize;
}

void TextureSynthesizerSampler::generate()
{
	computeWeightedMean(m_mean);
	return;
}

void TextureSynthesizerSampler::computeWeightedMean(ImageVectorType &mean)
{
	ERR_FAIL_COND_MSG(m_proceduralSampling.sampler() == nullptr,
					  "the sampler must be activated first (either with computeAutocovarianceSampler or set_cyclostationaryPeriods).");
	if(!m_exemplar.is_initialized())
	{
		computeImageVector();
	}
	ERR_FAIL_COND_MSG(!m_proceduralSampling.exemplarPtr() || !m_proceduralSampling.exemplarPtr()->is_initialized(),
					  "normal must be set with set_normal first.");
	m_proceduralSampling.computeWeightedMean(mean, m_exemplar.get_width(), m_exemplar.get_height(), m_meanAccuracy);
}

void TextureSynthesizerSampler::computeAutocovarianceSampler()
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
}

void TextureSynthesizerSampler::meanToComponent(Ref<Image> image, TextureTypeFlag component)
{
	ERR_FAIL_COND_MSG(image.is_null(), "image must not be null.");
	ERR_FAIL_COND_MSG(!m_mean.is_initialized(), "mean must be initialized.");
	outputImageToComponent(m_mean, component, image);
}

void TextureSynthesizerSampler::samplerPdfToImage(Ref<Image> image)
{
	ERR_FAIL_COND_MSG(!image->is_empty(), "image must be empty.");
	const TexSyn::SamplerImportance *si = dynamic_cast<const TexSyn::SamplerImportance *>(m_proceduralSampling.sampler());
	ERR_FAIL_COND_MSG(si == nullptr, "importance sampler must be set with computeAutocovarianceSampler().");
	Ref<Image> refPdf;
	refPdf = Image::create_empty(si->importanceFunction().get_width(), si->importanceFunction().get_height(), false, Image::FORMAT_RF);
	si->importanceFunction().toImage(refPdf, 0);
	image->copy_from(refPdf);
}

void TextureSynthesizerSampler::samplerRealizationToImage(Ref<Image> image, unsigned int size)
{
	ERR_FAIL_COND_MSG(!image->is_empty(), "image must be empty.");
	Ref<Image> refRealization;
	refRealization = Image::create_empty(size, 1, false, Image::FORMAT_RGF);
	ImageVectorType realization;
	m_proceduralSampling.preComputeSamplerRealization(realization, size);
	realization.toImage(refRealization);
	image->copy_from(refRealization);
}

void TextureSynthesizerSampler::centerExemplar(Ref<Image> exemplar, Ref<Image> mean)
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

void TextureSynthesizerSampler::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("set_cyclostationaryPeriods", "t0", "t1"), &TextureSynthesizerSampler::setCyclostationaryPeriods);
	ClassDB::bind_method(D_METHOD("set_importancePDF", "image"), &TextureSynthesizerSampler::setImportancePDF);
	ClassDB::bind_method(D_METHOD("set_meanAccuracy", "accuracy"), &TextureSynthesizerSampler::setMeanAccuracy);
	ClassDB::bind_method(D_METHOD("set_meanSize", "size"), &TextureSynthesizerSampler::setMeanSize);
	
	ClassDB::bind_method(D_METHOD("generate"), &TextureSynthesizerSampler::generate);
	
	ClassDB::bind_method(D_METHOD("samplerRealizationToImage", "image", "size"), &TextureSynthesizerSampler::samplerRealizationToImage, DEFVAL(4096));
	ClassDB::bind_method(D_METHOD("centerExemplar", "exemplar", "mean"), &TextureSynthesizerSampler::centerExemplar);
	ClassDB::bind_method(D_METHOD("computeAutocovarianceSampler"), &TextureSynthesizerSampler::computeAutocovarianceSampler);
	ClassDB::bind_method(D_METHOD("samplerPdfToImage", "image"), &TextureSynthesizerSampler::samplerPdfToImage);
}

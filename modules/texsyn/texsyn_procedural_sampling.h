#ifndef TEXSYN_PROCEDURAL_SAMPLING_H
#define TEXSYN_PROCEDURAL_SAMPLING_H

#include "image_vector.h"
#include "texsyn_sampler.h"

namespace TexSyn
{

template<typename T>
class ProceduralSampling
{
public:
	using DataType					= T;
	using ImageType					= ImageVector<DataType>;
	using PixelPosType				= Vector2i;
	//using PcaType						= PCA<DataType>;
	//using GaussianTransferType		= Gaussian_transfer<PcaImageType>;
	using LutType					= ImageType;
	using PtrImageType				= ImageScalar<void*>;
	using Vec2						= Vector2;

	ProceduralSampling();
	~ProceduralSampling();

	void set_exemplar(const ImageType *texture);
	void set_sampler(SamplerBase *sampler);

	const ImageType *exemplarPtr() const;
	const SamplerBase *sampler() const;

	void weightedMean(ImageType &mean, int width, int height, int nbSamples);
	void preComputeSamplerRealization(ImageVector<float> &realization, int size);

private:

	const ImageType						*m_exemplar;
	SamplerBase							*m_sampler;
};

template<typename T>
ProceduralSampling<T>::ProceduralSampling() :
	m_exemplar(),
	m_sampler(nullptr)
{}

template<typename T>
ProceduralSampling<T>::~ProceduralSampling()
{
	if(m_sampler != nullptr)
		memdelete(m_sampler);
}

template<typename T>
void ProceduralSampling<T>::set_exemplar(const ImageType *texture)
{
	m_exemplar = texture;
}

template<typename T>
void ProceduralSampling<T>::set_sampler(SamplerBase *sampler)
{
	m_sampler = sampler;
}

template<typename T>
const typename ProceduralSampling<T>::ImageType *ProceduralSampling<T>::exemplarPtr() const
{
	return m_exemplar;
}

template<typename T>
const SamplerBase *ProceduralSampling<T>::sampler() const
{
	return m_sampler;
}

template<typename T>
void ProceduralSampling<T>::weightedMean(ImageType &mean, int width, int height, int nbSamples)
{
	DEV_ASSERT(m_sampler != nullptr && m_exemplar!=nullptr && m_exemplar->is_initialized());
	mean.init(width, height, m_exemplar->get_nbDimensions(), true);
	SamplerBase::VectorType offsets;
	m_sampler->generate(offsets, nbSamples);
	mean.parallel_for_all_images([&] (typename ImageType::ImageScalarType &image, int index)
	{
		image.for_all_pixels([&] (typename ImageType::DataType &pix, int x, int y)
		{
			for(const Vec2 &offset : offsets)
			{
				double xd=double(x)/(mean.get_width()-1) + offset.x;
				double yd=double(y)/(mean.get_height()-1) + offset.y;
				xd = xd-Math::floor(xd);
				yd = yd-Math::floor(yd);
				pix += m_exemplar->get_pixelInterp(xd, yd, index);
			}
			pix = pix * (1.0/nbSamples);
		});
	});
}

template<typename T>
void ProceduralSampling<T>::preComputeSamplerRealization(ImageVector<float> &realization, int size)
{
	DEV_ASSERT(m_sampler != nullptr);
	realization.init(size, 1, 2);
	SamplerBase::VectorType offsets;
	m_sampler->generate(offsets, size);
	realization.parallel_for_all_images([&] (ImageScalar<float> &image, int index)
	{
		image.for_all_pixels([&] (typename ImageType::DataType &pix, int x, int y)
		{
			const Vec2 &offset = offsets[x];
			pix=offset[index];
		});
	});
}

}

#endif

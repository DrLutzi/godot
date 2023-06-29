#include "wasserstein.h"

namespace TexSyn
{

double DistanceFunction::l2Distance(const ImageType &im1, const ImageType &im2)
{
	double distance = 0.0;
    im1.for_all_images([&] (const ImageScalarType &sc1, int index)
	{
		double distanceImage = 0.0;
		const ImageScalarType &sc2 = im2.get_image(index);
		sc1.for_all_pixels([&] (const ImageType::DataType &pix1, int x, int y)
		{
			double xd = double(x)/im1.get_width();
			double yd = double(y)/im1.get_height();
			ImageType::DataType pix2 = sc2.get_pixelInterp(xd, yd);
			double pixelDistance = sqrt((pix1 - pix2)*(pix1 - pix2));
			distanceImage += pixelDistance;
		});
		distanceImage /= im1.get_width()*im1.get_height();
		distance += distanceImage;
	});
	distance /= 3.0;
	return distance;
}

Wasserstein::Wasserstein() :
    m_distanceFunction(),
    m_data(),
    m_sinkhornNbIterations(50),
    m_sinkhornRegularization(0.5)
{}

void Wasserstein::addTexture(const ImageType &image, double weight)
{
    TextureAndWeight tw;
	tw.image = image;
	tw.interpolationWeight = weight;
    m_data.push_back(tw);
	return;
}

void Wasserstein::setWeight(unsigned int i, double weight)
{
	DEV_ASSERT(i<m_data.size());
	m_data[i].interpolationWeight = weight;
	return;
}

void Wasserstein::setDistanceFunction(const DistanceFunctionType &function)
{
	m_distanceFunction = function;
}

void Wasserstein::computeOptimalTransportPlan()
{

}

};

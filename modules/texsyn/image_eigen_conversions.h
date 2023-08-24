#ifndef TEXSYN_IMAGE_EIGEN_H
#define TEXSYN_IMAGE_EIGEN_H

#include "image_vector.h"
#include "eigen/Eigen/Dense"
#include "mipmaps.h"

namespace TexSyn
{

///
/// \brief fromImageVectorToMatrix fills an eigen matrix of size (nb of texels, nb of dimensions) from an image
/// \param image the image to base the matrix on
/// \param matrix the matrix to fill
///
template<typename T>
void fromImageVectorToMatrix(const ImageVector<T> &image, Eigen::MatrixXd &matrix)
{
	//Assuming image has the correct dimensions
	int i=0;
	image.for_all_images([&] (const typename ImageVector<T>::ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (const typename ImageVector<T>::DataType &pix)
		{
			matrix(j, i) = pix;
			++j;
		});
		++i;
	});
}

template<typename T>
void fromMatrixToImageVector(const Eigen::MatrixXd &matrix, ImageVector<T> &image)
{
	//Assuming image has the correct dimensions
	int i=0;
	image.for_all_images([&] (typename ImageVector<T>::ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (typename ImageVector<T>::DataType &pix)
		{
			pix = matrix(j, i);
			++j;
		});
		++i;
	});
}

///
/// \brief fromImageVectorToMatrixWithRegionID fills an eigen matrix of size (nb of texels, nb of dimensions) from an image, but only if the region of that texel is id.
/// \param image the image to base the matrix on
/// \param matrix the matrix to fill
/// \param regionIDMap the map giving all ids
/// \param id the id to use
/// \return the number of regions with this ID
///
template<typename T>
int fromImageVectorToMatrixWithRegionID(const ImageVector<T> &image, Eigen::MatrixXd &matrix, const ImageScalar<BitVector> &regionIDMap, uint64_t id)
{
	//Assuming image has the correct dimensions
	int nbTexels = 0;
	int i=0;
	image.for_all_images([&] (const typename ImageVector<T>::ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (const typename ImageVector<T>::DataType &pix, int x, int y)
		{
			ImageScalar<BitVector>::DataType region = regionIDMap.get_pixel(x, y);
			if((region & id).toBool())
			{
				matrix(j, i) = pix;
				++j;
				if(i == 0)
				{
					++nbTexels;
				}
			}
		});
		++i;
	});
	return nbTexels;
}

template<typename T>
int fromMatrixToImageVectorWithRegionID(const Eigen::MatrixXd &matrix, ImageVector<T> &image, const ImageScalar<BitVector> &regionIDMap, uint64_t id)
{
	//Assuming image has the correct dimensions
	int nbTexels = 0;
	int i=0;
	image.for_all_images([&] (typename ImageVector<T>::ImageScalarType &scalar)
	{
		int j=0;
		scalar.for_all_pixels([&] (typename ImageVector<T>::DataType &pix, int x, int y)
		{
			ImageScalar<BitVector>::DataType region = regionIDMap.get_pixel(x, y);
			if((region & id).toBool())
			{
				pix = matrix(j, i);
				++j;
				if(i == 0)
				{
					++nbTexels;
				}
			}
		});
		++i;
	});
	return nbTexels;
}

};

#endif

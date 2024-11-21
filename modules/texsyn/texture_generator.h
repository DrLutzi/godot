#ifndef TEXSYN_TEXTURE_GENERATOR_H
#define TEXSYN_TEXTURE_GENERATOR_H

#include "texture_synthesizer.h"
#include "core/math/vector2i.h"

namespace TexSyn
{

class TextureGenerator : public RefCounted
{
	GDCLASS(TextureGenerator, RefCounted);

public:

	struct Gradient
	{
		Color color1;
		Color color2;
	};

	using ImageVectorType = TexSyn::ImageVector<float>;
	using ImageScalarType = TexSyn::ImageScalar<float>;
	using ImageRegionType = TexSyn::ImageScalar<int>;

	TextureGenerator();
	
	void setWidth(int width);
	void setHeight(int height);
	
	void generateRandomGradiantGrid(int cellSize);
	void generateGridIDMap(int cellSize);
	
	void generateRandomGradientCellular(int cellSize);
	void generateCellularIDMap(int cellSize);
	
	void toImage(Ref<Image> refImage) const;
	
protected:

	Gradient getRandomGradient(int seed) const;

	static void _bind_methods();
	
private:

	ImageVectorType m_image;
	int m_width;
	int m_height;
};

}
#endif

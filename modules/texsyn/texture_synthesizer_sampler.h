#ifndef SAMPLERTEXTURESYNTHESIZER_H
#define SAMPLERTEXTURESYNTHESIZER_H

#include "texture_synthesizer.h"

class TextureSynthesizerSampler : public TextureSynthesizer
{
	GDCLASS(TextureSynthesizerSampler, TextureSynthesizer);

public:

	TextureSynthesizerSampler();

	void setCyclostationaryPeriods(Vector2 t0, Vector2 t1);
	void setImportancePDF(Ref<Image> image);
	void setMeanAccuracy(unsigned int accuracy);
	void setMeanSize(unsigned int meanSize);
	
	void generate();

	void meanToComponent(Ref<Image> image, TextureTypeFlag component);
	void samplerPdfToImage(Ref<Image> image);
	void samplerRealizationToImage(Ref<Image> image, unsigned int size);
	void centerExemplar(Ref<Image> exemplar, Ref<Image> mean);
	
protected:
	static void _bind_methods();
	
private:
	void computeWeightedMean(ImageVectorType &mean);
	void computeAutocovarianceSampler();

	TexSyn::ProceduralSampling<float> m_proceduralSampling;
	unsigned int m_meanAccuracy;
	unsigned int m_meanSize;
	ImageVectorType m_mean;
	
};

#endif // SAMPLERTEXTURESYNTHESIZER_H

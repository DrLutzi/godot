#ifndef CONTENTATLAS_H
#define CONTENTATLAS_H

#include "mipmaps.h"
#include <queue>

class ContentAtlas
{
public:

	using MipmapType = TexSyn::Mipmap;
	using MipmapMultiIDType = TexSyn::MipmapMultiIDMap;
	using ImageMultiIDType = MipmapMultiIDType::ImageMultiIDMapType;
	using ImageVectorType = MipmapType::ImageType;
	using ImageScalarType = ImageVectorType::ImageScalarType;
	using PosType = ImageScalarType::PosType;
	using AtlasIndexType = PosType;
	
	struct BoundingBox
	{
		PosType origin;
		int width, height;
	};

	struct ReducedContent
	{
		BoundingBox bbox;
		ImageScalarType hitbox;
		unsigned int region;
		unsigned int mipmapLevel;
	};
	
	class CompareReducedContents
	{
	public:
		CompareReducedContents(){}
		bool operator()(const ReducedContent &object, const ReducedContent &other)
		{
			return object.bbox.height < other.bbox.height;
		}
	};

	ContentAtlas();
	
	void setContributions(const LocalVector<MipmapType> &contributions);
	void setRegionMap(const ImageMultiIDType &region, unsigned int nbRegions);
	void setContentsFromAllVersions(const LocalVector<ImageVectorType> &contents);
	
	void computeAllAtlases();
	
	const LocalVector<ImageVectorType> &atlasVector() const;
	ImageVectorType originTexture() const; //< constructs and returns a compact representation of m_origins.
	
private:

	void computeOriginsAndEmplacePlan();
	void computeOneAtlas(ImageVectorType &atlas, int contentID);

	ReducedContent findReducedContentOfRegionAtResolution(int region, int mipmapLevel);
	void computeEmplacePlan();
	
	bool emplacePlan_checkEmplace(const ReducedContent &reducedContent, int x, int y);
	void emplacePlan_findEmplace(const ReducedContent &reducedContent, int &x, int &y);
	void emplacePlan_doEmplace(const ReducedContent &reducedContent, int x, int y);

	LocalVector<LocalVector<MipmapType>> m_contents; //< stores the contents, isolated, and mipmapped. First dimension: id. Second dimension: version of the content.
	LocalVector<MipmapType> m_contributions; //< stores the contributions of each region, mipmapped.
	MipmapMultiIDType m_regionMap; //< all regions
	unsigned int m_nbRegions;
	LocalVector<LocalVector<PosType>> m_origins; //< origins in the atlas
	LocalVector<ImageVectorType> m_atlasVector; //< final image vectors
	ImageScalarType m_occupationMap; //< occupation map
	
	std::priority_queue<ReducedContent, std::vector<ReducedContent>, CompareReducedContents> m_reducedContentQueue;
	HashMap<AtlasIndexType, PosType> m_emplacePlan; //< plan to know which content to put where in the atlas.
	HashMap<AtlasIndexType, BoundingBox> m_boundingBoxes; //< remembers bounding boxes after the emplace plan is computed.
	
};

#endif // CONTENTATLAS_H

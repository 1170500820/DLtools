

default_plm = 'bert-base-chinese'


# dataset
NYT_relations = [
    "/business/company/founders",
    "/people/person/place_of_birth",
    "/people/deceased_person/place_of_death",
    "/business/company_shareholder/major_shareholder_of",
    "/people/ethnicity/people",
    "/location/neighborhood/neighborhood_of",
    "/sports/sports_team/location",
    "/business/company/industry",
    "/business/company/place_founded",
    "/location/administrative_division/country",
    "/sports/sports_team_location/teams",
    "/people/person/nationality",
    "/people/person/religion",
    "/business/company/advisors",
    "/people/person/ethnicity",
    "/people/ethnicity/geographic_distribution",
    "/business/person/company",
    "/business/company/major_shareholders",
    "/people/person/place_lived",
    "/people/person/profession",
    "/location/country/capital",
    "/location/location/contains",
    "/location/country/administrative_divisions",
    "/people/person/children",
    # "None"
]
NYT_relations_idx = {x: i for (i, x) in enumerate(NYT_relations)}

duie_relations = [
    "毕业院校",
    "嘉宾",
    "配音",
    "主题曲",
    "代言人",
    "所属专辑",
    "父亲",
    "作者",
    "上映时间",
    "母亲",
    "专业代码",
    "占地面积",
    "邮政编码",
    "票房",
    "注册资本",
    "主角",
    "妻子",
    "编剧",
    "气候",
    "歌手",
    "获奖",
    "校长",
    "创始人",
    "首都",
    "丈夫",
    "朝代",
    "饰演",
    "面积",
    "总部地点",
    "祖籍",
    "人口数量",
    "制片人",
    "修业年限",
    "所在城市",
    "董事长",
    "作词",
    "改编自",
    "出品公司",
    "导演",
    "作曲",
    "主演",
    "主持人",
    "成立日期",
    "简称",
    "海拔",
    "号",
    "国籍",
    "官方语言", ]
duie_relations_idx = {x: i for (i, x) in enumerate(duie_relations)}


WebNLG_relations = [
 "senators" ,
 "postalCode" ,
 "founder" ,
 "jurisdiction" ,
 "demonym" ,
 "headquarter" ,
 "partsType" ,
 "abbreviation" ,
 "course" ,
 "manager" ,
 "almaMater" ,
 "rector" ,
 "patronSaint" ,
 "alternativeNames" ,
 "youthclub" ,
 "mayor" ,
 "occupation" ,
 "dishVariation" ,
 "alternativeName" ,
 "IATA_Location_Identifier" ,
 "languages" ,
 "affiliations" ,
 "areaOfWater" ,
 "ethnicGroup" ,
 "mascot" ,
 "ingredient" ,
 "division" ,
 "significantBuilding" ,
 "ethnicGroups" ,
 "chancellor" ,
 "mainIngredients" ,
 "material" ,
 "fat" ,
 "isPartOf" ,
 "has to its southeast" ,
 "champions" ,
 "impactFactor" ,
 "was given the 'Technical Campus' status by" ,
 "administrativeArrondissement" ,
 "child" ,
 "dedicatedTo" ,
 "formerName" ,
 "EISSN_number" ,
 "precededBy" ,
 "academicDiscipline" ,
 "4th_runway_LengthFeet" ,
 "architecture" ,
 "publisher" ,
 "served" ,
 "river" ,
 "numberOfPostgraduateStudents" ,
 "numberOfRooms" ,
 "dean" ,
 "creator" ,
 "series" ,
 "crewMembers" ,
 "numberOfStudents" ,
 "dateOfRetirement" ,
 "related" ,
 "placeOfDeath" ,
 "affiliation" ,
 "bedCount" ,
 "frequency" ,
 "backup pilot" ,
 "elevationAboveTheSeaLevel" ,
 "year" ,
 "distributor" ,
 "served as Chief of the Astronaut Office in" ,
 "numberOfMembers" ,
 "category" ,
 "established" ,
 "added to the National Register of Historic Places" ,
 "nearestCity" ,
 "was a crew member of" ,
 "was selected by NASA" ,
 "religion" ,
 "state" ,
 "ISBN_number" ,
 "fossil" ,
 "capital" ,
 "currentTenants" ,
 "governingBody" ,
 "developer" ,
 "buildingType" ,
 "leader" ,
 "1st_runway_Number" ,
 "None" ,
 "officialLanguage" ,
 "leaderParty" ,
 "runwaySurfaceType" ,
 "inaugurationDate" ,
 "lastAired" ,
 "birthName" ,
 "transportAircraft" ,
 "awards" ,
 "has to its northwest" ,
 "starring" ,
 "address" ,
 "genre" ,
 "areaOfLand" ,
 "motto" ,
 "has to its north" ,
 "fullname" ,
 "tenant" ,
 "legislature" ,
 "numberOfPages" ,
 "compete in" ,
 "neighboringMunicipality" ,
 "language" ,
 "2nd_runway_SurfaceType" ,
 "nationality" ,
 "country" ,
 "yearOfConstruction" ,
 "sportsGoverningBody" ,
 "bird" ,
 "gemstone" ,
 "1st_runway_LengthFeet" ,
 "architecturalStyle" ,
 "genus" ,
 "completionDate" ,
 "leaderName" ,
 "creatorOfDish" ,
 "ground" ,
 "hometown" ,
 "municipality" ,
 "birthDate" ,
 "LCCN_number" ,
 "similarDish" ,
 "UTCOffset" ,
 "sportsOffered" ,
 "operator" ,
 "academicStaffSize" ,
 "OCLC_number" ,
 "nativeName" ,
 "has to its southwest" ,
 "attackAircraft" ,
 "city" ,
 "family" ,
 "district" ,
 "timeInSpace" ,
 "areaCode" ,
 "servingSize" ,
 "followedBy" ,
 "regionServed" ,
 "floorCount" ,
 "numberOfUndergraduateStudents" ,
 "populationTotal" ,
 "leaderTitle" ,
 "editor" ,
 "ISSN_number" ,
 "runwayLength" ,
 "influencedBy" ,
 "location" ,
 "cityServed" ,
 "doctoralAdvisor" ,
 "birthPlace" ,
 "3rd_runway_LengthFeet" ,
 "league" ,
 "club" ,
 "foundingDate" ,
 "officialSchoolColour" ,
 "award" ,
 "chairmanTitle" ,
 "firstAppearanceInFilm" ,
 "part" ,
 "firstAired" ,
 "representative" ,
 "countySeat" ,
 "commander" ,
 "placeOfBirth" ,
 "foundedBy" ,
 "parentCompany" ,
 "3rd_runway_SurfaceType" ,
 "aircraftFighter" ,
 "cost" ,
 "architect" ,
 "deathPlace" ,
 "residence" ,
 "latinName" ,
 "voice" ,
 "was awarded" ,
 "region" ,
 "largestCity" ,
 "website" ,
 "ReferenceNumber in the National Register of Historic Places" ,
 "CODEN_code" ,
 "owner" ,
 "significantProject" ,
 "outlookRanking" ,
 "carbohydrate" ,
 "currency" ,
 "4th_runway_SurfaceType" ,
 "5th_runway_SurfaceType" ,
 "governmentType" ,
 "hubAirport" ,
 "protein" ,
 "notableWork" ,
 "servingTemperature" ,
 "LibraryofCongressClassification" ,
 "elevationAboveTheSeaLevel_(in_feet)" ,
 "locationIdentifier" ,
 "buildingStartDate" ,
 "administrativeCounty" ,
 "aircraftHelicopter" ,
 "owningOrganisation" ,
 "mediaType" ,
 "battles" ,
 "campus" ,
 "5th_runway_Number" ,
 "chairman" ,
 "season" ,
 "floorArea" ,
 "populationDensity" ,
 "has to its northeast" ,
 "literaryGenre" ,
 "higher" ,
 "status" ,
 "product" ,
 "designer" ,
 "firstPublicationYear" ,
 "ICAO_Location_Identifier" ,
 "foundationPlace" ,
 "runwayName" ,
 "1st_runway_LengthMetre" ,
 "director" ,
 "1st_runway_SurfaceType" ,
 "keyPerson" ,
 "president" ,
 "fullName" ,
 "deathDate" ,
 "nickname" ,
 "class" ,
 "spokenIn" ,
 "operatingOrganisation" ,
 "chairperson" ,
 "author" ,
 "broadcastedBy" ,
 "headquarters" ,
 "height" ,
 "elevationAboveTheSeaLevel_(in_metres)" ,
 "chief" ,
 "locationCity" ,
 "anthem" ,
 "title" ,
 "order" ,
 "areaTotal" ,
 "has to its west"
]
WebNLG_relations_idx = {x: i for (i, x) in enumerate(WebNLG_relations)}



# model


# train
plm_lr = 2e-5
others_lr = 1e-4

default_bsz = 8
default_shuffle = True

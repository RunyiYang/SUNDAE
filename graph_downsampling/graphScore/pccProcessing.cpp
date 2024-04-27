// Copyright (C) 2018, 2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory.h>
#include <sys/stat.h>

#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

#include "pccProcessing.hpp"

using namespace std;
using namespace pcc_processing;

// trim from start
static inline std::string &ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  std::not1(std::ptr_fun<int, int>(std::isspace))));
  return s;
}

// trim from end
static inline std::string &rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
  return s;
}

// trim from both ends
static inline std::string &trim(std::string &s) {
  return ltrim(rtrim(s));
}

/**!
 * **************************************
 *  Class PointXYZSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

PointXYZSet::~PointXYZSet()
{
}

void
PointXYZSet::init( long int size, int i0, int i1, int i2 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;
  if (i1 >= 0)
    idxInLine[1] = i1;
  if (i2 >= 0)
    idxInLine[2] = i2;

  p.resize( size );
  for (long int i = 0; i < size; i++)
    p[i].resize( 3 );
}

/**!
 * \brief
 *  loadPoints
 *
 * \param out
 *  return 0, if succeed
 *  return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
PointXYZSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT32 )
  {
    for (int i = 0; i < 3; i++)
      p[idx][i] = *(float*)(pPcc->lineMem + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT64 )
  {
    for (int i = 0; i < 3; i++)
      p[idx][i] = *(double*)(pPcc->lineMem + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else
  {
    cerr << "Error! Wrong xyz data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * \brief
 *  loadPoints
 *
 * \param out
 *  return 0, if succeed
 *  return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
PointXYZSet::loadPoints( PccPointCloud *pPcc, void* val, long int idx )
{
  // pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT64
  for (int i = 0; i < 3; i++)
    p[idx][i] = *( (double*)val + i );
  return 0;
}

/**!
 * **************************************
 *  Class RGBSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

/**!
 * \brief
 *  RGBSet
 *
 *  Dong Tian <tian@merl.com>
 */
RGBSet::~RGBSet()
{
}

void
RGBSet::init( long int size, int i0, int i1, int i2 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;
  if (i1 >= 0)
    idxInLine[1] = i1;
  if (i2 >= 0)
    idxInLine[2] = i2;

  c.resize( size );
  for (long int i = 0; i < size; i++)
    c[i].resize( 3 );
}

int
RGBSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT8 )
  {
    for (int i = 0; i < 3; i++)
      c[idx][i] = *(unsigned char*)(pPcc->lineMem + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else
  {
    cerr << "Error! Wrong rgb data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * \brief
 *  loadPoints
 *
 * \param out
 *  return 0, if succeed
 *  return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
RGBSet::loadPoints( PccPointCloud *pPcc, void* val, long int idx )
{
  // pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT8
  for (int i = 0; i < 3; i++)
    c[idx][i] = *( (unsigned char*)val + i );
  return 0;
}

/**!
 * **************************************
 *  Class NormalSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

NormalSet::~NormalSet()
{
}

void
NormalSet::init( long int size, int i0, int i1, int i2 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;
  if (i1 >= 0)
    idxInLine[1] = i1;
  if (i2 >= 0)
    idxInLine[2] = i2;

  n.resize( size );
  for (long int i = 0; i < size; i++)
    n[i].resize( 3 );
}

/**!
 * \brief
 *   loadPoints
 *
 * \param out
 *   return 0, if succeed
 *   return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
NormalSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT32 )
  {
    for (int i = 0; i < 3; i++)
      n[idx][i] = *(float*)(pPcc->lineMem + pPcc->fieldPos[ idxInLine[i] ]);
  }
  else
  {
    cerr << "Error! Wrong xyz data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * \brief
 *  loadPoints
 *
 * \param out
 *  return 0, if succeed
 *  return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
NormalSet::loadPoints( PccPointCloud *pPcc, void* val, long int idx )
{
  // pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::FLOAT32
  for (int i = 0; i < 3; i++)
    n[idx][i] = *( (double*)val + i );
  return 0;
}

/**!
 * **************************************
 *  Class LidarSet
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

LidarSet::~LidarSet()
{
}

void
LidarSet::init( long int size, int i0 )
{
  if (i0 >= 0)
    idxInLine[0] = i0;

  reflectance.resize( size );
}

int
LidarSet::loadPoints( PccPointCloud *pPcc, long int idx )
{
  if ( pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT16 )
  {
    reflectance[ idx ] = *(unsigned short*)(pPcc->lineMem + pPcc->fieldPos[ idxInLine[0] ]);
  }
  else
  {
    cerr << "Error! Wrong reflectance data type: " << pPcc->fieldType[ idxInLine[0] ] << endl;
    return -1;
  }
  return 0;
}

/**!
 * \brief
 *  loadPoints
 *
 * \param out
 *  return 0, if succeed
 *  return non-zero, otherwise
 *
 *  Dong Tian <tian@merl.com>
 */
int
LidarSet::loadPoints( PccPointCloud *pPcc, void* val, long int idx )
{
  // pPcc->fieldType[ idxInLine[0] ] == PccPointCloud::PointFieldTypes::UINT16
  reflectance[idx] = *(unsigned short*) val;
  return 0;
}

/**!
 * **************************************
 *  Class PccPointCloud
 *    To load point clouds
 *
 *  Dong Tian <tian@merl.com>
 * **************************************
 */

PccPointCloud::PccPointCloud()
{
  size = 0;
  fileFormat = -1;
  memset( fieldType, 0, sizeof(int)*MAX_NUM_FIELDS );
  memset( fieldPos,  0, sizeof(int)*MAX_NUM_FIELDS );
  fieldNum = 0;
  dataPos = 0;
  lineNum = 0;

  bXyz = bRgb = bNormal = bLidar = false;
}

PccPointCloud::~PccPointCloud()
{
}

/**!
 * \brief
 *  checkFile
 *
 * \param out
 * return the position of the field, if found
 * return -1 if not found
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::checkFile(string fileName)
{
  ifstream in;
  string line;
  string str[3];
  int counter = 0;

  int secIdx = 0;               // 1: vertex

  bool bPly = false;
  bool bEnd = false;
  fieldSize = 0;                  // position in the line memory buffer

  in.open(fileName, ifstream::in);
  while ( getline(in, line) && lineNum < 100) // Maximum number of lines of header section: 100
  {
    line = rtrim( line );
    if (line == "ply")
    {
      bPly = true;
    }
    else
    {
      counter = 0;
      stringstream ssin(line);
      while (ssin.good() && counter < 3){
        ssin >> str[counter];
        counter++;
      }

      if ( str[0] == "format" )
      {
        if ( str[1] == "ascii" )
          fileFormat = 0;
        else if ( str[1] == "binary_little_endian" )
          fileFormat = 1;
        else
        {
          cerr << "Format not to be handled: " << line << endl;
          return -1;
        }
      }
      else if ( str[0] == "element" )
      {
        if (str[1] == "vertex")
        {
          size = stol( str[2] );
          secIdx = 1;
        }
        else
          secIdx = 2;
      }
      else if ( str[0] == "property" && secIdx == 1 && str[1] != "list" )
      {
        fieldPos[fieldNum] = fieldSize;
        if ( str[1] == "uint8" || str[1] == "uchar" ) {
          fieldType[fieldNum] = PointFieldTypes::UINT8;
          fieldSize += sizeof(unsigned char);
        }
        else if ( str[1] == "int8" || str[1] == "char" ) {
          fieldType[fieldNum] = PointFieldTypes::INT16;
          fieldSize += sizeof(char);
        }
        else if ( str[1] == "uint16" || str[1] == "ushort" ) {
          fieldType[fieldNum] = PointFieldTypes::UINT16;
          fieldSize += sizeof(unsigned short);
        }
        else if ( str[1] == "int16" || str[1] == "short" ) {
          fieldType[fieldNum] = PointFieldTypes::INT16;
          fieldSize += sizeof(short);
        }
        else if ( str[1] == "uint" || str[1] == "uint32" ) {
          fieldType[fieldNum] = PointFieldTypes::UINT32;
          fieldSize += sizeof(unsigned int);
        }
        else if ( str[1] == "int" || str[1] == "int32" ) {
          fieldType[fieldNum] = PointFieldTypes::INT32;
          fieldSize += sizeof(int);
        }
        else if ( str[1] == "float" || str[1] == "float32" ) {
          fieldType[fieldNum] = PointFieldTypes::FLOAT32;
          fieldSize += sizeof(float);
        }
        else if ( str[1] == "float64" || str[1] == "double" ) {
          fieldType[fieldNum] = PointFieldTypes::FLOAT64;
          fieldSize += sizeof(double);
        }
        else {
          cerr << "Incompatible header section. Line: " << line << endl;
          bPly = false;
        }
        fieldNum++;
      }
      else if ( str[0] == "end_header" )
      {
        lineNum++;
        bEnd = true;
        break;
      }
    }

    lineNum++;
  }

  dataPos = in.tellg(); // Only to be used under Linux. @DT
  in.close();

  if (bPly && fileFormat >= 0 && fieldNum > 0 && size > 0 && bEnd)
    return 0;

  cout << "Warning: The header section seems not fully compatible!\n";
  return -1;
}

/*
 * \brief
 *   Check the attribute by parsing the header section
 *
 * \param[in]
 *   fileName: Input point cloud file
 *   fieldName: Attribute name
 *   fieldTypeX: Potential data type of the attribute
 *
 * \return -1, if fails
 *         index of this attribute in a row, if succeeds
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::checkField(string fileName, string fieldName, string fieldType1, string fieldType2, string fieldType3, string fieldType4)
{
  ifstream in;
  string line;
  string str[3];
  int counter = 0;
  bool bFound = false;
  int iPos = -1;

  int secIdx = 0;               // 1: vertex.
  in.open(fileName, ifstream::in);

  while ( getline(in, line) )
  {
    if (line == "ply")
      ;
    else
    {
      // cout << line << endl;
      counter = 0;
      stringstream ssin(line);
      while (ssin.good() && counter < 3){
        ssin >> str[counter];
        counter++;
      }

      if ( str[0] == "format" )
        ;                       // do nothing
      else if ( str[0] == "element" )
      {
        if (str[1] == "vertex")
          secIdx = 1;
        else
          secIdx = 2;
      }
      else if ( str[0] == "property" && secIdx == 1 && str[1] != "list" )
      {
        iPos++;
        if ( str[1] == fieldType1 || str[1] == fieldType2 || str[1] == fieldType3 || str[1] == fieldType4 )
        {
          if ( str[2] == fieldName )
          {
            bFound = true;
            break;
          }
        }
      }
    }
  }
  in.close();

  if ( !bFound )
    iPos = -1;

  return iPos;
}

/*
 * \brief
 *   Load one line of data from the input file
 *
 * \param[in]
 *   in: Input file stream
 *
 * \return 0, if succeed
 *         non-zero, if failed
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::loadLine( ifstream &in )
{
  int i;
  if (fileFormat == 0)          // ascii
  {
    for (i = 0; i < fieldNum; i++)
    {
      switch (fieldType[i])
      {
      case PointFieldTypes::UINT8:
        in >> ((unsigned short*)(lineMem+fieldPos[i]))[0]; // @DT: tricky. not using uint8 when reading
        break;
      case PointFieldTypes::INT8:
        in >> ((short*)(lineMem+fieldPos[i]))[0];
        break;
      case PointFieldTypes::UINT16:
        in >> ((unsigned short*)(lineMem+fieldPos[i]))[0];
        break;
      case PointFieldTypes::INT16:
        in >> ((short*)(lineMem+fieldPos[i]))[0];
        break;
      case PointFieldTypes::UINT32:
        in >> ((unsigned int*)(lineMem+fieldPos[i]))[0];
        break;
      case PointFieldTypes::INT32:
        in >> ((int*)(lineMem+fieldPos[i]))[0];
        break;
      case PointFieldTypes::FLOAT32:
        in >> ((float*)(lineMem+fieldPos[i]))[0];
        break;
      case PointFieldTypes::FLOAT64:
        in >> ((double*)(lineMem+fieldPos[i]))[0];
        break;
      default:
        cout << "Unknown field type: " << fieldType[i] << endl;
        return -1;
        break;
      }
    }
  }
  else if (fileFormat == 1)     // binary_little_endian
  {
    for (i = 0; i < fieldNum; i++)
    {
      switch (fieldType[i])
      {
      case PointFieldTypes::UINT8:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(unsigned char));
        break;
      case PointFieldTypes::INT8:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(char));
        break;
      case PointFieldTypes::UINT16:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(unsigned short));
        break;
      case PointFieldTypes::INT16:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(short));
        break;
      case PointFieldTypes::UINT32:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(unsigned int));
        break;
      case PointFieldTypes::INT32:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(int));
        break;
      case PointFieldTypes::FLOAT32:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(float));
        break;
      case PointFieldTypes::FLOAT64:
        in.read((char*)(lineMem+fieldPos[i]), sizeof(double));
        break;
      default:
        cout << "Unknown field type: " << fieldType[i] << endl;
        return -1;
        break;
      }
    }
  }
  else
    return -1;

  return 0;
}

#if _WIN32
int
PccPointCloud::seekAscii(ifstream &in)
{
  string line;
  int lineNum = 0;
  bool bFound = false;
  while (!bFound && lineNum < 1000 && getline(in, line))
  {
    if (line == "end_header")
      bFound = true;
  }
  if (bFound)
    return 0;
  return -1;
}

int
PccPointCloud::seekBinary(ifstream &in)
{
  // Move the pointer to the beginning of the data
  // in.seekg( dataPos );
  char buf[200];
  char mark[] = "end_header\n";
  int  markLen = sizeof(mark);
  int posFile = 0;
  int posBuf = 0;
  while (strcmp(buf, mark) != 0 && posFile < 9000)
  {
    in.read(&buf[posBuf], 1);
    buf[posBuf + 1] = 0;
    if (posBuf >= markLen - 1)
    {
      memcpy(buf, buf + 1, markLen + 1);
      buf[markLen - 1] = 0;
    }
    else
      posBuf++;
    posFile++;
  }
  if (posFile >= 9000)
    return -1;
  return 0;
}
#endif


/*
 * \brief load data into memory
 * \param[in]
 *   inFile: File name of the input point cloud
 *   isNormal: true to import normals. false to load non-normal data
 *
 * \return 0, if succeed
 *         non-zero, if failed
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::load( string inFile, bool isNormal )
{
  int ret = 0;

  int iPos[3];                  // The position of x,y,z in the list

  if (inFile == "")
    return 0;

  struct stat bufferExist;
  if (stat (inFile.c_str(), &bufferExist) != 0)
  {
    cout << "File does not exist: " << inFile << endl;
    return -1;
  }

  // Check file header
  if ( checkFile( inFile ) != 0 )
    return -1;

  if ( !isNormal )              // Load non-normal data
  {
    // Make sure (x,y,z) available and determine the float type
    iPos[0] = checkField( inFile, "x", "float", "float32", "float64", "double" );
    iPos[1] = checkField( inFile, "y", "float", "float32", "float64", "double" );
    iPos[2] = checkField( inFile, "z", "float", "float32", "float64", "double" );
    if ( iPos[0] < 0 && iPos[1] < 0 && iPos[2] < 0 )
      return -1;
    xyz.init(size, iPos[0], iPos[1], iPos[2]);
    bXyz = true;

    // Make sure (r,g,b) available and determine the float type
    iPos[0] = checkField( inFile, "red",   "uint8", "uchar" );
    iPos[1] = checkField( inFile, "green", "uint8", "uchar" );
    iPos[2] = checkField( inFile, "blue",  "uint8", "uchar" );
    if ( iPos[0] >= 0 && iPos[1] >= 0 && iPos[2] >= 0 )
    {
      rgb.init(size, iPos[0], iPos[1], iPos[2]);
      bRgb = true;
    }

    // Make sure (lidar) available and determine the float type
    iPos[0] = checkField( inFile, "reflectance", "uint16" );
    if ( iPos[0] >= 0 )
    {
      lidar.init(size, iPos[0]);
      bLidar = true;
    }

    // Load regular data (rather than normals)
    ifstream in;

    if (fileFormat == 0)
    {
      in.open(inFile, ifstream::in);
#if _WIN32
      if (seekAscii(in) != 0)
      {
        cout << "Check the file header section. Incompatible header!" << endl;
        in.close();
        return -1;
      }
#else
      in.seekg(dataPos);
#endif
    }
    else if (fileFormat == 1)
    {
      in.open(inFile, ifstream::in | ifstream::binary);
#if _WIN32
      if (seekBinary(in) != 0)
      {
        cout << "Check the file header section. Incompatible header!" << endl;
        in.close();
        return -1;
      }
#else
      in.seekg(dataPos);
#endif
    }

    for (long int i = 0; i < size; i++)
    {
      if ( loadLine( in ) < 0 )   // Load the data into line memory
      {
        ret = -1;
        break;
      }
      xyz.loadPoints( this, i );
      if (bRgb)
        rgb.loadPoints( this, i );
      if (bLidar)
        lidar.loadPoints( this, i );
    }

    in.close();

#if 1
    {
      long int i = size - 1;
      cout << "Verifying if the data is loaded correctly.. The last point is: ";
      cout << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << endl;
    }
#endif

#if 0
    {
      ofstream outF;
      outF.open( "testItPly.ply", ofstream::out );
      for (long int i = 0; i < size; i++)
      {
        outF << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << " " << (unsigned short) rgb.c[i][0] << " " << (unsigned short) rgb.c[i][1] << " " << (unsigned short) rgb.c[i][2] << " " << lidar.reflectance[i] << endl;
        // outF << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << " " << (unsigned short) rgb.c[i][0] << " " << (unsigned short) rgb.c[i][1] << " " << (unsigned short) rgb.c[i][2] << endl;
        // outF << xyz.p[i][0] << " " << xyz.p[i][1] << " " << xyz.p[i][2] << endl;
      }
      outF.close();
    }
#endif
  }

  else                          // load normal
  {
    // Make sure (x,y,z) available and determine the float type
    iPos[0] = checkField( inFile, "nx", "float", "float64", "double" );
    iPos[1] = checkField( inFile, "ny", "float", "float64", "double" );
    iPos[2] = checkField( inFile, "nz", "float", "float64", "double" );
    if ( iPos[0] < 0 && iPos[1] < 0 && iPos[2] < 0 )
      return -1;
    normal.init(size, iPos[0], iPos[1], iPos[2]);
    bNormal = true;

    if (bNormal)
    {
      // Load normal data
      ifstream in;

      if (fileFormat == 0)
      {
        in.open(inFile, ifstream::in);
#if _WIN32
        if (seekAscii(in) != 0)
        {
          cout << "Check the file header section. Incompatible header!" << endl;
          in.close();
          return -1;
        }
#else
        in.seekg(dataPos);
#endif
      }
      else if (fileFormat == 1)
      {
        in.open(inFile, ifstream::in | ifstream::binary);
#if _WIN32
        if (seekBinary(in) != 0)
        {
          cout << "Check the file header section. Incompatible header!" << endl;
          in.close();
          return -1;
        }
#else
        in.seekg(dataPos);
#endif
      }

      for (long int i = 0; i < size; i++)
      {
        if ( loadLine( in ) < 0 )   // Load the data into line memory
        {
          ret = -1;
          break;
        }
        normal.loadPoints( this, i );
      }

      in.close();
    }
  }
  return ret;
}


/*
 * \brief load data into memory
 * \param[in]
 *   inFile: File name of the input point cloud
 *   isNormal: true to import normals. false to load non-normal data
 *
 * \return 0, if succeed
 *         non-zero, if failed
 *
 *  Dong Tian <tian@merl.com>
 */
int
PccPointCloud::loadBlock( double *blk, long int sz, bool isNormal )
{
  int ret = 0;

  size = sz;

  if (!isNormal)
  {
    xyz.init(size, 0, 1, 2);    // use the 0, 1, and 2 position to store x, y and z
    fieldType[ 0 ] = PccPointCloud::PointFieldTypes::FLOAT64;
    fieldType[ 1 ] = PccPointCloud::PointFieldTypes::FLOAT64;
    fieldType[ 2 ] = PccPointCloud::PointFieldTypes::FLOAT64;
    bXyz = true;

    for (long int i = 0; i < size; i++)
    {
      double aPoint[3];
      for (int j = 0; j < 3; j++)
        aPoint[j] = blk[i + j*size];

      xyz.loadPoints( this, aPoint, i );
    }
  }

  else                          // normals
  {
    normal.init(size, 0, 1, 2);    // use the 0, 1, and 2 position to store x, y and z
    fieldType[ 0 ] = PccPointCloud::PointFieldTypes::FLOAT64;
    fieldType[ 1 ] = PccPointCloud::PointFieldTypes::FLOAT64;
    fieldType[ 2 ] = PccPointCloud::PointFieldTypes::FLOAT64;
    bNormal = true;

    for (long int i = 0; i < size; i++)
    {
      double aPoint[3];
      for (int j = 0; j < 3; j++)
        aPoint[j] = blk[i + j*size];

      normal.loadPoints( this, aPoint, i );
    }
  }

  return ret;
}

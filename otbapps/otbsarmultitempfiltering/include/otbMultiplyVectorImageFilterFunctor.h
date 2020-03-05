/*=========================================================================

  Program:   ORFEO Toolbox
  Language:  C++
  Date:      $Date$
  Version:   $Revision$


  Copyright (c) Centre National d'Etudes Spatiales. All rights reserved.
  See OTBCopyright.txt for details.


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef otbMultiplyVectorImageFilterFunctor_h
#define otbMultiplyVectorImageFilterFunctor_h



namespace otb
{
namespace Functor
{

template<class TInput1, class TInput2, class TOutputPixel>
class MultiplyVectorImageFilterFunctor
{
public:
  inline TOutputPixel operator ()(const TInput1& p1, const TInput2& p2)
  {
    unsigned int nbComp = p1.GetSize();
    TOutputPixel outValue(nbComp);
    for(unsigned int i = 0; i<nbComp;++i)
      {
      outValue[i] = static_cast<typename TOutputPixel::ValueType>(p1[i]*p2[i]/p2[1]);

      }
    return outValue;
  }
};
}
} // end namespace otb

#endif

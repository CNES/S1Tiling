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
#ifndef otbOutcoreFunctor_h
#define otbOutcoreFunctor_h
#include <iostream>


namespace otb
{

/** \class Functor::OutcoreFunctor
 *
 * - compute the outcore for the multitemp filter
 *

 * \ingroup Functor
 *
 * \ingroup OTBMultiTempFilter
 */
namespace Functor
{

template<class TInput, class TOutput>
class OutcoreFunctor
{
public:
  OutcoreFunctor() {}
  virtual ~OutcoreFunctor() {}
  inline TOutput operator ()(const TInput& itA)
  {
    TOutput meanA;
    meanA.SetSize(1);
    meanA.Fill(0.);
    TOutput pixel;
    pixel.SetSize(1);

    TOutput nb_non_zero_pixels;
    nb_non_zero_pixels.SetSize(1);
    nb_non_zero_pixels.Fill(0.);
    
    for (unsigned long pos = 0; pos < itA.Size(); ++pos)
      {
         pixel = itA.GetPixel(pos);
         meanA += pixel;
         for (unsigned long i = 0; i < pixel.Size(); ++i)
            {
               if (pixel[i] != 0.)
                  {
                     nb_non_zero_pixels[i]++;
                  }
            }
      }
    for (unsigned long i = 0; i < pixel.Size(); ++i)
        {
           if (nb_non_zero_pixels[i] !=0.)
             {
                meanA[i] /= nb_non_zero_pixels[i];
             }
           else
             {
                meanA[i]=0.;
             }
        }

    TOutput ratio=itA.GetCenterPixel();
    ratio.SetSize(meanA.GetSize()+1);   // add an element to store the ENL (nb of images used for temporal average)

    for (unsigned long i = 0; i < meanA.Size(); ++i)
       {
	   if ((meanA[i]!=0.)&&(ratio[i]!=0.))
	   {
		   ratio[i]/=meanA[i];
	   }
	   else
	   {
		   ratio[i]=0.;
	   }
       // Compute the ENL (nb of images used for temporal average)
       ratio[meanA.Size()]=int(ratio[0]>0.);
       }
    return static_cast<TOutput>(ratio);
  }
};
}
} // end namespace otb

#endif

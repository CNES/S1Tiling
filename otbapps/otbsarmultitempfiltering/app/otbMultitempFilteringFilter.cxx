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
#include "otbWrapperApplication.h"
#include "otbWrapperApplicationFactory.h"
#include "otbMultiplyVectorImageFilter.h"
#include "otbImageList.h" 
#include "otbMultiplyVectorImageFilter.h"
#include "otbMeanFilter.h"
#include "otbImageFileWriter.h"
#include "otbMultiToMonoChannelExtractROI.h"
#include <iostream>

namespace otb
{

namespace Wrapper
{

class MultitempFilteringFilter : public Application
{
public:
  /** Standard class typedefs. */
  typedef MultitempFilteringFilter   Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;
  typedef ImageList<FloatVectorImageType>    ImageListType;
  typedef typename otb::MultiplyVectorImageFilter<FloatVectorImageType,FloatVectorImageType,FloatVectorImageType> MultiplyVectorImageFilterType;
  typedef otb::ImageFileWriter<FloatVectorImageType> WriterType;
  typedef otb::MultiToMonoChannelExtractROI<FloatVectorImageType::InternalPixelType,
                                            FloatVectorImageType::InternalPixelType> ExtractChannelFilterType;

  /** Standard macro */
  itkNewMacro(Self);

  itkTypeMacro(MultitempFilteringFilter, otb::Application);

  typedef MeanFilter<FloatVectorImageType,FloatVectorImageType> MeanFilterType;

private:
  void DoInit()
  {
    SetName("MultitempFilteringFilter");
    SetDescription("");

    // Documentation
    SetDocName("MultitempFilteringFilter");
    SetDocLongDescription("This application implement the Quegan speckle filter for SAR images. It applies the outcore to a list of images. The outcore is genenerated by the MultitempFilteringOutcore application" );
    
						  
    SetDocLimitations("None");
    SetDocAuthors("Thierry Koleck (CNES)");
    SetDocSeeAlso("MultitempFilteringOutcore");

    AddDocTag(Tags::SAR);

    AddParameter(ParameterType_InputImageList,  "inl",   "Input images list");
    SetParameterDescription("inl", "Input image list");

    AddParameter(ParameterType_Int ,  "wr",   "Spatial averaging Window radius ");
    SetParameterDescription("wr", "Window radius");
    
    AddParameter(ParameterType_InputImage, "oc",  "Outcore filename");
    SetParameterDescription("oc", "Outcore filename");

    AddParameter(ParameterType_OutputImage, "enl",  "ENL filename");
    SetParameterDescription("enl", "Number of images averaged");
       
    AddRAMParameter();

  }

  void DoUpdateParameters()
  {
    // Nothing to do here : all parameters are independent
  }

  void DoExecute()
  {
    int Radius = this->GetParameterInt("wr");
    FloatVectorImageListType::Pointer inList = this->GetParameterImageList("inl");
    // On verifie que la liste en entree n'est pas vide
    std::cout << inList->Size() << "\n";
    if( inList->Size() == 0 )
      {
      itkExceptionMacro("No input Image set...");
      }
    std::vector< std::string> filelist;
    filelist=this->GetParameterStringList("inl");

    inList->GetNthElement(0)->UpdateOutputInformation();
    // Recupere la taille de l'image

    // Initialise le filtre de calcul du outcore

    FloatVectorImageType::Pointer Outcore = GetParameterFloatVectorImage("oc");

    // Calcul des images de sortie
    m_MultiplyByOutcoreImageFilter=MultiplyVectorImageFilterType::New();

    m_MeanImageFilter=MeanFilterType::New();
    m_MultiplyByOutcoreImageFilter=MultiplyVectorImageFilterType::New();


    for( unsigned int i=0; i<inList->Size(); i++ )
      {
      m_MeanImageFilter->SetInput(inList->GetNthElement(i));
      m_MeanImageFilter->SetRadius(Radius);
      m_MeanImageFilter->UpdateOutputInformation();

      m_MultiplyByOutcoreImageFilter->SetInput1(m_MeanImageFilter->GetOutput());
      m_MultiplyByOutcoreImageFilter->SetInput2(Outcore);
      m_MultiplyByOutcoreImageFilter->UpdateOutputInformation();

      // Definit le nom du fichier de sortie (images filtrees)
      std::ostringstream oss;
      if(filelist[i].find("/") != std::string::npos)
      {
	      size_t lastindex = filelist[i].find_last_of("."); 
	      size_t lastindex2 = filelist[i].find_last_of("/"); 

	      oss << filelist[i].substr(0,lastindex2)<<"/filtered"<<filelist[i].substr(lastindex2,lastindex-lastindex2) << "_filtered" <<filelist[i].substr(lastindex);
      } else {
		  size_t lastindex = filelist[i].find_last_of(".");

	      oss << "filtered"<<filelist[i].substr(0,lastindex) << "_filtered" <<filelist[i].substr(lastindex);
      }

      // writer label
      WriterType::Pointer writer = WriterType::New();
      writer->SetFileName(oss.str());
      writer->SetInput(m_MultiplyByOutcoreImageFilter->GetOutput());
      AddProcess(writer, writer->GetFileName());
      writer->Update();
      }

      m_Filter = ExtractChannelFilterType::New();
      m_Filter->SetInput(Outcore);
      m_Filter->SetChannel(Outcore->GetVectorLength());
      m_Filter->UpdateOutputInformation();
      SetParameterOutputImage("enl", m_Filter->GetOutput());
      SetParameterOutputImagePixelType("enl",ImagePixelType_uint16);
      
  }

  MultiplyVectorImageFilterType::Pointer m_MultiplyByOutcoreImageFilter;   
  MeanFilterType::Pointer           m_MeanImageFilter;
  ExtractChannelFilterType::Pointer m_Filter;
}; 

} //end namespace Wrapper
} //end namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::MultitempFilteringFilter)

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
#include "otbOutcoreFilter.h"
#include "otbMultiplyVectorImageFilter.h"
#include "itkAddImageFilter.h"
#include "otbImage.h" 
#include "otbImageList.h" 
#include "itkBinaryFunctorImageFilter.h"
#include <itkVariableLengthVector.h>
#include "otbMultiplyVectorImageFilter.h"
#include "otbObjectList.h"
#include "otbMeanFilter.h"
#include "otbImageFileWriter.h"

//#include "boost/filesystem/path.hpp"

namespace otb
{

namespace Wrapper
{

class MultitempFiltering : public Application
{
public:
  /** Standard class typedefs. */
  typedef MultitempFiltering   Self;
  typedef Application                         Superclass;
  typedef itk::SmartPointer<Self>             Pointer;
  typedef itk::SmartPointer<const Self>       ConstPointer;
  typedef ImageList<FloatVectorImageType>    ImageListType;
  typedef itk::AddImageFilter<FloatVectorImageType,FloatVectorImageType,FloatVectorImageType> AddImageFilterType;
  typedef typename otb::MultiplyVectorImageFilter<FloatVectorImageType,FloatVectorImageType,FloatVectorImageType> MultiplyVectorImageFilterType;
  typedef otb::ImageFileWriter<FloatVectorImageType> WriterType;


  /** Standard macro */
  itkNewMacro(Self);

  itkTypeMacro(MultitempFiltering, otb::Application);

  typedef OutcoreFilter<FloatVectorImageType,FloatVectorImageType> OutcoreFilterType;
  typedef MeanFilter<FloatVectorImageType,FloatVectorImageType> MeanFilterType;

private:
  void DoInit()
  {
    SetName("MultitempFiltering");
    SetDescription("");

    // Documentation
    SetDocName("MultitempFiltering");
    SetDocLongDescription("" );
    
						  
    SetDocLimitations("None");
    SetDocAuthors("Thierry Koleck (CNES)");
    SetDocSeeAlso("");

    AddDocTag(Tags::SAR);

    AddParameter(ParameterType_InputImageList,  "inl",   "Input images list");
    SetParameterDescription("inl", "Input image list");

    AddParameter(ParameterType_Int ,  "wr",   "Spatial averaging Window radius ");
    SetParameterDescription("wr", "Window radius");

    //AddParameter(ParameterType_OutputImage, "out",  "Output name");
    //SetParameterDescription("out", "Output name");
    
    AddRAMParameter();

    // Default values
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
    //boost::filesystem::path my_path(filelist[0]);
    //std::cout<< my_path.string();
    inList->GetNthElement(0)->UpdateOutputInformation();
    // Recupere la taille de l'image

    // Initialise le filtre de calcul du outcore
    

    m_AddImageFilter=AddImageFilterType::New();

    FilterList=otb::ObjectList<itk::ImageToImageFilter<FloatVectorImageType,FloatVectorImageType> >::New();
    OutcoreFilterType::Pointer m_OutcoreFilter;  
    m_OutcoreFilter=OutcoreFilterType::New();
    m_OutcoreFilter->SetInput(inList->GetNthElement(0));
    m_OutcoreFilter->SetRadius(Radius);
    m_OutcoreFilter->UpdateOutputInformation();
    FilterList->PushBack(m_OutcoreFilter);

    for( unsigned int i=1; i<inList->Size(); i++ ) {
       m_OutcoreFilter=OutcoreFilterType::New();
       m_OutcoreFilter->SetInput(inList->GetNthElement(i));
       m_OutcoreFilter->SetRadius(Radius);
       m_OutcoreFilter->UpdateOutputInformation();
       m_AddImageFilter=AddImageFilterType::New();
       m_AddImageFilter->SetInput1(m_OutcoreFilter->GetOutput());
       m_AddImageFilter->SetInput2(FilterList->GetNthElement(2*i-1)->GetOutput());
       m_AddImageFilter->UpdateOutputInformation();

       FilterList->PushBack(m_OutcoreFilter);
       FilterList->PushBack(m_AddImageFilter);
    }

    m_MultiplyByConstImageFilter=MultiplyVectorImageFilterType::New();
    m_MultiplyByConstImageFilter->SetInput(FilterList->Back()->GetOutput());
    itk::VariableLengthVector<float> constante=itk::VariableLengthVector<float>(1);
    constante.Fill(1./inList->Size());
    m_MultiplyByConstImageFilter->SetConstant(constante);
    
    FilterList->PushBack(m_MultiplyByConstImageFilter);

    // Calcul des images de sortie
    m_MultiplyByOutcoreImageFilter=MultiplyVectorImageFilterType::New();

    m_MeanImageFilter=MeanFilterType::New();
    WriterType::Pointer writer = WriterType::New();


    for( unsigned int i=0; i<inList->Size(); i++ )
      {
      m_MeanImageFilter->SetInput(inList->GetNthElement(i));
      //FloatVectorImageType::SizeType indexRadius;

      m_MeanImageFilter->SetRadius(Radius);
      m_MeanImageFilter->UpdateOutputInformation();

      m_MultiplyByOutcoreImageFilter=MultiplyVectorImageFilterType::New();
      m_MultiplyByOutcoreImageFilter->SetInput1(m_MeanImageFilter->GetOutput());
      m_MultiplyByOutcoreImageFilter->SetInput2(m_MultiplyByConstImageFilter->GetOutput());

      m_MultiplyByOutcoreImageFilter->UpdateOutputInformation();

      // Definit le nom du fichier de sortie (images filtrees)
      std::ostringstream oss;

      size_t lastindex = filelist[i].find_last_of("."); 
      size_t lastindex2 = filelist[i].find_last_of("/"); 
      std::string rawname = filelist[i].substr(0, lastindex); 
      oss << filelist[i].substr(0,lastindex2)<<"/filtered"<<filelist[i].substr(lastindex2,lastindex-lastindex2) << "_filtered" <<filelist[i].substr(lastindex);
     
      //OutputImageParameter::Pointer paramOut = OutputImageParameter::New();

      // writer label
      std::ostringstream osswriter;
      osswriter<< "writer (File : "<< i<<")";
/*
      // Set the filename of the current output image
      paramOut->SetFileName(oss.str());
      otbAppLogINFO(<< "File: "<<paramOut->GetFileName() << " will be written.");
      //paramOut->SetValue(m_MultiplyByOutcoreImageFilter->GetOutput());
      paramOut->SetValue(m_MeanImageFilter->GetOutput());
      //paramOut->SetValue(inList->GetNthElement(i));
      paramOut->SetPixelType(this->GetParameterOutputImagePixelType("out"));
      std::cout << this->GetParameterOutputImagePixelType("out") << "\n";
      // Add the current level to be written
      paramOut->InitializeWriters();
      AddProcess(paramOut->GetWriter(), osswriter.str());

      paramOut->Write();
*/      
      writer->SetFileName(oss.str());
      writer->SetInput(m_MultiplyByOutcoreImageFilter->GetOutput());
      otbAppLogINFO(<< "File: "<<writer->GetFileName() << " will be written.");
      AddProcess(writer, osswriter.str());

      writer->Update();

      }
    //SetParameterOutputImage("out", m_MultiplyByOutcoreImageFilter->GetOutput());
  }
  otb::ObjectList<itk::ImageToImageFilter<FloatVectorImageType,FloatVectorImageType> >::Pointer FilterList;

  AddImageFilterType::Pointer m_AddImageFilter; 
  MultiplyVectorImageFilterType::Pointer m_MultiplyByOutcoreImageFilter;   
  MultiplyVectorImageFilterType::Pointer m_MultiplyByConstImageFilter;  
  MeanFilterType::Pointer           m_MeanImageFilter;
  
}; 

} //end namespace Wrapper
} //end namespace otb

OTB_APPLICATION_EXPORT(otb::Wrapper::MultitempFiltering)

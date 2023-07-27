// ---------------------------------------------------------
//
// History:
//
// Created by Karim Massri (karim.massri@cern.ch) 2017-01-25
// Updated by Nicolas Lurkin (nicolas.lurkin@cern.ch) 2018-11-27
//
// ---------------------------------------------------------

#ifndef KMU3SELECTION_HH
#define KMU3SELECTION_HH

#include "Analyzer.hh"
#include "Pi0Selection.hh"
#include "SpectrometerRICHAssociationAlgorithm.hh"
#include "VertexLSF.hh"

class Kmu3Selection: public NA62Analysis::Analyzer {

public:
  explicit Kmu3Selection(NA62Analysis::Core::BaseAnalysis *ba);
  ~Kmu3Selection();
  void InitHist();
  void InitOutput();
  void Process(Int_t);
  void StartOfBurstUser() {}
  void EndOfBurstUser() {}
  void StartOfRunUser() {}
  void EndOfRunUser() {}
  void EndOfJobUser();
  void BuildPDFReport();
  void PostProcess() {}
  void DrawPlot() {}
  void PrintStatisticsPerBurst();

private:
  Int_t fTriggerMask;   ///< Definition of the data sample by L0 trigger mask
  Bool_t fReadingData;  ///< Reading data or my own output?
  Bool_t
    fSkipWrongType;  ///< If true, do not process MC events not of Kmu3 type (used by automatic revision metrics)
  Bool_t fGTKEnabled;    ///< Do we use GTK with additional selection
  Double_t fMaxNBursts;  ///< Number of bins in the histograms of counts vs burst ID, default = 5000

  TH1F *fHZtrue;  ///< Histogram of the true Zvertex (filled for MC from KineParts)
  TH1F *fHPhysicsEventsPerBurst;
  TH1F *fHKmu3EventsPerBurst;
  TH1F *fHMass;
  TH1F *fHEOP;
  TH1F *fHPtot;
  TH1F *fHPttot;
  TH1F *fHZvtx;
  TH2F *fHMassVsMomentum;
  VertexLSF fVertexLSF;

  // Outputs
  Bool_t fEventSelected;
  Double_t fKmu3Time;
  Int_t fKmu3TrackID;
  Pi0SelectionOutput fPi0SelectionOutput;
};
#endif

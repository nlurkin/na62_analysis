// This analyzer is a clone of Kmu3Selection, where the selection cuts for the electron track are updated for muon:
//  - Request inside MUV3 acceptance
//  - Request MUV3 association instead of no association
//  - Use muon mass hypothesis instead of electron
//  - Skip positron E/p studies

#include "Kmu3Selection.hh"

#include "BeamParameters.hh"
#include "DownstreamTrack.hh"
#include "GeometricAcceptance.hh"
#include "KaonDecayConstants.hh"
#include "LAVMatching.hh"
#include "SAVMatching.hh"
#include "TriggerConditions.hh"

#include <TCanvas.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TStyle.h>

#include <stdlib.h>

using namespace NA62Analysis;

Kmu3Selection::Kmu3Selection(Core::BaseAnalysis *ba): Analyzer(ba, "Kmu3Selection"), fHZtrue(nullptr) {
  RequestTree("LKr", new TRecoLKrEvent, "Reco");
  RequestTree("LAV", new TRecoLAVEvent, "Reco");
  RequestTree("IRC", new TRecoIRCEvent, "Reco");
  RequestTree("SAC", new TRecoSACEvent, "Reco");
  RequestL0Data();

  fReadingData = kTRUE;

  AddParam("TriggerMask", &fTriggerMask, 0xFF);
  AddParam("MaxNBursts", &fMaxNBursts, 5000);  // max number of bins in histograms
  AddParam("SkipWrongType", &fSkipWrongType, false);

  fHPhysicsEventsPerBurst = nullptr;
  fHKmu3EventsPerBurst = nullptr;
  fHMass = nullptr;
  fHEOP = nullptr;
  fHPtot = nullptr;
  fHPttot = nullptr;
  fHZvtx = nullptr;
  fHMassVsMomentum = nullptr;
  fEventSelected = false;
  fKmu3Time = 0.0;
  fKmu3TrackID = -1;
  fPi0SelectionOutput.fPi0Momentum = TLorentzVector();
  fPi0SelectionOutput.fTime = 0.0;
  fPi0SelectionOutput.fPosition = TVector3();
  fPi0SelectionOutput.fKaonMomentum = TLorentzVector();
  fPi0SelectionOutput.fGammaMomenta.first = TLorentzVector();
  fPi0SelectionOutput.fGammaMomenta.second = TLorentzVector();
  fPi0SelectionOutput.fClustersID.first = -1;
  fPi0SelectionOutput.fClustersID.second = -1;
}

Kmu3Selection::~Kmu3Selection() {
}

void Kmu3Selection::InitHist() {
  fReadingData = GetIsTree();  // false in --histo mode, true otherwise

  if(fReadingData) {
    if(GetWithMC()) {
      BookHisto("mctrue/hZvertex", new TH1F("Zvertex_true", "True Zvertex; z [m]", 300, 0, 300));
    }

    // LKr-related quantities
    BookHisto(
      "LKr/hNLKrCells",
      new TH1F("NLKrCells", "Number of LKr cells with signal;Number of cells", 125, -0.5, 249.5));
    BookHisto("LKr/hNLKrClusters",
              new TH1F("NLKrClusters", "Number of LKr clusters;Number of clusters", 10, -0.5, 9.5));
    BookHisto("LKr/hLKrNAssociatedClusters",
              new TH1F("LKrNAssociatedClusters", "Number of LKr clusters associated to a track", 4,
                       -0.5, 3.5));
    BookHisto("LKr/hLKrDDeadCell",
              new TH1F("LKrDDeadCell",
                       "Track distance to nearest dead cell;Distance to deal cell [mm]", 200, 0,
                       4000));
    BookHisto("LKr/hLKrClusterEnergy",
              new TH1F("LKrClusterEnergy", "LKr cluster energy;Energy [GeV]", 200, 0, 100));
    BookHisto("LKr/hLKrClusterTime",
              new TH1F("LKrClusterTime", "LKr cluster time wrt trigger;Time [ns]", 200, -50, 50));
    BookHisto("LKr/hLKrCellTotalEnergy",
              new TH1F("LKrCellTotalEnergy",
                       "LKr total cell (E>40MeV) energy;Total cell energy [GeV]", 70, 0, 70));
    BookHisto("LKr/hLKrCellClusterTotalEnergy",
              new TH2F("LKrCellClusterTotalEnergy",
                       "LKr total cluster energy vs cell energy;Total cell (>40MeV) energy "
                       "[GeV];Total cluster energy [GeV]",
                       100, 0, 100, 100, 0, 100));

    // Positron E/p studies
    BookHisto("LKr/hLKrEoP", new TH1F("LKrEoP", "Positron E/p;Track E/p", 150, 0.0, 1.5));
    BookHisto("LKr/hLKrEoPVsMomentum",
              new TH2F("LKrEoPVsMomentum",
                       "Positron E/p vs momentum;Track momentum [GeV/c];Track E/p", 60, 0, 60, 150,
                       0.0, 1.5));

    BookHisto("LKr/hLKrEoPVsMomentum_RICH_pion",
              new TH2F("LKrEoPVsMomentum_RICH_pion",
                       "Positron E/p vs momentum;Track momentum [GeV/c];Track E/p", 60, 0, 60, 150,
                       0.0, 1.5));
    BookHisto("LKr/hLKrEoPVsMomentum_RICH_pion_mod",
              new TH2F("LKrEoPVsMomentum_RICH_pion_mod",
                       "Positron E/p vs momentum;Track momentum [GeV/c];Track E/p", 60, 0, 60, 150,
                       0.0, 1.5));
    BookHisto("LKr/hNHitsPVsMomentum_RICH_pion",
              new TH2F("NHitsPVsMomentum_RICH_pion",
                       "N(associated RICH hits) vs momentum;Track momentum [GeV/c];Number of hits",
                       60, 0, 60, 30, -0.5, 29.5));

    BookHisto("LKr/hLKrEoPVsMomentum_RICH_electron",
              new TH2F("LKrEoPVsMomentum_RICH_electron",
                       "Positron E/p vs momentum;Track momentum [GeV/c];Track E/p", 60, 0, 60, 150,
                       0.0, 1.5));
    BookHisto("LKr/hLKrEoPVsMomentum_RICH_electron_mod",
              new TH2F("LKrEoPVsMomentum_RICH_electron_mod",
                       "Positron E/p vs momentum;Track momentum [GeV/c];Track E/p", 60, 0, 60, 150,
                       0.0, 1.5));
    BookHisto("LKr/hNHitsPVsMomentum_RICH_electron",
              new TH2F("NHitsPVsMomentum_RICH_electron",
                       "N(associated RICH hits) vs momentum;Track momentum [GeV/c];Number of hits",
                       60, 0, 60, 30, -0.5, 29.5));

    BookHisto("LKr/hRichHypothesisVsLKrEoP",
              new TH2F("RichHypothesisVsLKrEoP",
                       "Positron RICH hypothesis vs E/p;Track E/p;RICH hypothesis", 150, 0.0, 1.5,
                       6, -1.5, 4.5));
    BookHisto("LKr/hLKrEoP_DistDeadCell",
              new TH1F("LKrEoP_DistDeadCell", "Positron E/p;Positron E/p", 150, 0.0, 1.5));
    BookHisto(
      "LKr/hLKrEoPVsMomentum_DistDeadCell",
      new TH2F("LKrEoPVsMomentum_DistDeadCell",
               "Positron E/p vs momentum [D(DeadCell)>20mm];Positron momentum [GeV/c];Positron E/p",
               40, 0, 80, 150, 0.0, 1.5));

    // General histograms
    BookHisto("hNTracks", new TH1F("hNTracks", "Number of tracks", 11, -0.5, 10.5));
    BookHisto("hNGoodTracks", new TH1F("hNGoodTracks", "Number of good tracks", 11, -0.5, 10.5));
    BookHisto("hNInTimeTracks", new TH1F("hNInTimeTracks", "Number of tracks", 11, -0.5, 10.5));
    BookHisto("hTrackTime", new TH1F("hTrackTime", "Track time", 200, -50, 50));
    BookHisto("hClusterEnergy", new TH1F("hClusterEnergy", "Cluster energy", 200, 0, 50));
    BookHisto("hClusterTime", new TH1F("hClusterTime", "Cluster time", 200, -50, 50));
    BookHisto("hClusterDistance", new TH1F("hClusterDistance", "hClusterDistance", 200, 0, 2000));
    BookHisto("hNPhotonCandidates",
              new TH1F("NPhotonCandidates", "NPhotonCandidates", 11, -0.5, 10.5));
    BookHisto("hdZChargedNeutralVertex",
              new TH1F("hdZChargedNeutralVertex", "hdZChargedNeutralVertex", 200, -40, 40));

    BookHisto("hEOP", new TH1F("hEOP", "Track E/p; E/p", 150, 0.0, 1.5));
    BookHisto("hZvtx",
              new TH1F("hZvtx", "Z of track-beam axis vertex;Vertex z [m]", 100, 100, 200));
    BookHisto("hCDA", new TH1F("hCDA", "CDA of the track-beam axis vertex;CDA [mm]", 200, 0, 200));
    BookHisto("hPtot", new TH1F("hPtot", "Total momentum;P_{tot} [GeV/c]", 200, 0., 100.));
    BookHisto("hPttot",
              new TH1F("hPttot", "Total transverse momentum;P_{T}^{tot} [MeV/c]", 250, 0., 500.));
    BookHisto(
      "hMMiss2_Kmu3",
      new TH1F("hMMiss2_Kmu3",
               "Squared missing mass in Kmu3 hypothesis;M_{miss}^{2}(K_{e3}) [GeV^{2}/c^{4}]", 600,
               -0.15, 0.15));
    BookHisto("hPMMiss2_Kmu3", new TH2F("hPMMiss2_Kmu3",
                                       "Squared missing mass in Kmu3 hypothesis vs momentum;Track "
                                       "momentum [GeV/c];M_{miss}^{2}(K_{e3}) [GeV^{2}/c^{4}]",
                                       70, 0, 70, 300, -0.15, 0.15));
    BookHisto(
      "hMMiss2_K2pi",
      new TH1F("hMMiss2_K2pi",
               "Squared missing mass in K2pi hypothesis;M_{miss}^{2}(K_{2#pi}) [GeV^{2}/c^{4}]",
               600, -0.15, 0.15));
    BookHisto("hPMMiss2_K2pi", new TH2F("hPMMiss2_K2pi",
                                        "Squared missing mass in K2pi hypothesis vs momentum;Track "
                                        "momentum [GeV/c];M_{miss}^{2}(K_{2#pi}) [GeV^{2}/c^{4}]",
                                        70, 0, 70, 300, -0.15, 0.15));
    BookHisto(
      "hMMiss2_El",
      new TH1F("hMMiss2_El",
               "Squared missing mass in e^{+} hypothesis;M_{miss}^{2}(e^{+}) [GeV^{2}/c^{4}]", 600,
               -0.15, 0.15));
    BookHisto("hPMMiss2_El", new TH2F("hPMMiss2_El",
                                      "Squared missing mass in e^{+} hypothesis vs momentum;Track "
                                      "momentum [GeV/c];M_{miss}^{2}(e^{+}) [GeV^{2}/c^{4}]",
                                      70, 0, 70, 300, -0.15, 0.15));
    BookHisto(
      "hMMiss2_Pi",
      new TH1F("hMMiss2_Pi",
               "Squared missing mass in #pi^{+} hypothesis;M_{miss}^{2}(#pi^{+}) [GeV^{2}/c^{4}]",
               600, -0.15, 0.15));
    BookHisto("hPMMiss2_Pi",
              new TH2F("hPMMiss2_Pi",
                       "Squared missing mass in #pi^{+} hypothesis vs momentum;Track momentum "
                       "[GeV/c];M_{miss}^{2}(#pi^{+}) [GeV^{2}/c^{4}]",
                       70, 0, 70, 300, -0.15, 0.15));
    BookHisto("hMass2_Kmu3",
              new TH1F("hMass2_Kmu3", "Squared mass in Kmu3 hypothesis;M^{2}(K_{e3}) [GeV^{2}/c^{4}]",
                       600, 0.0, 0.3));
    BookHisto("hPMass2_Kmu3", new TH2F("hPMass2_Kmu3",
                                      "Squared mass in Kmu3 hypothesis vs momentum;Track momentum "
                                      "[GeV/c];M^{2}(K_{e3}) [GeV^{2}/c^{4}]",
                                      70, 0, 70, 300, 0.0, 0.3));
    BookHisto("hMass2_K2pi",
              new TH1F("hMass2_K2pi",
                       "Squared mass in K2pi hypothesis;M^{2}(K_{2#pi}) [GeV^{2}/c^{4}]", 600, 0.0,
                       0.3));
    BookHisto("hPMass2_K2pi", new TH2F("hPMass2_K2pi",
                                       "Squared mass in K2pi hypothesis vs momentum;Track  "
                                       "momentum [GeV/c];M^{2}(K_{2#pi}) [GeV^{2}/c^{4}]",
                                       70, 0, 70, 300, 0.0, 0.3));

    BookHisto(
      "hPTheta",
      new TH2F("hPTheta",
               "Track opening angle wrt beam axis vs momentum;Track momentum [GeV/c];#theta", 160,
               0, 80, 100, 0.0, 0.02));
    BookHisto("hMMiss2_Kmu3_Final",
              new TH1F("hMMiss2_Kmu3_Final", "hMMiss2_Kmu3_Final", 600, -0.15, 0.15));
    BookHisto("hPMMiss2_Kmu3_Final",
              new TH2F("hPMMiss2_Kmu3_Final",
                       "Squared missing mass in Kmu3 hypothesis vs momentum;Track momentum "
                       "[GeV/c];M_{miss}^{2}(K_{e3}) [GeV^{2}/c^{4}]",
                       70, 0, 70, 600, -0.15, 0.15));

    BookHisto("hPhysicsEventsPerBurst",
              new TH1F("hPhysicsEventsPerBurst", "Physics events per burst;Burst ID", fMaxNBursts,
                       -0.5, fMaxNBursts - 0.5));
    BookHisto("hKmu3EventsPerBurst",
              new TH1F("hKmu3EventsPerBurst", "Kmu3 candidates per burst;Burst ID", fMaxNBursts, -0.5,
                       fMaxNBursts - 0.5));
    BookHisto("hKmu3Time", new TH1F("hKmu3Time", "Kmu3 event time;Kmu3 time [ns]", 800, -200., 200.));
  }

  else {  // step 2
    std::cout << user_normal() << "Reading my own output" << std::endl;
    fHZtrue = static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "mctrue/Zvertex_true", true));
    fHPhysicsEventsPerBurst =
      static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hPhysicsEventsPerBurst", true));
    fHKmu3EventsPerBurst =
      static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hKmu3EventsPerBurst", true));
    fHMass = static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hMMiss2_Kmu3", true));
    fHZvtx = static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hZvtx", true));
    fHPtot = static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hPtot", true));
    fHPttot = static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hPttot", true));
    fHEOP = static_cast<TH1F *>(RequestHistogram(fAnalyzerName, "hEOP", true));
    fHMassVsMomentum = static_cast<TH2F *>(RequestHistogram(fAnalyzerName, "hPMMiss2_Kmu3", true));
  }
}

void Kmu3Selection::InitOutput() {
  RegisterOutput("EventSelected", &fEventSelected);
  RegisterOutput("Kmu3Time", &fKmu3Time);
  RegisterOutput("Kmu3TrackID", &fKmu3TrackID);
  RegisterOutput("Kmu3Pi0SelectionOutput", &fPi0SelectionOutput);
}

void Kmu3Selection::Process(Int_t) {

  if(!fReadingData)
    return;  // no action if reading its own output in --histo mode
  if(GetWithMC() && fSkipWrongType && GetMCEvent()->GetEventBoundary(0)->GetStreamID() % 1000 != 42)
    return;  // Not a Kmu3 event

  SetOutputState("EventSelected", kOValid);
  SetOutputState("Kmu3Time", kOValid);
  SetOutputState("Kmu3TrackID", kOValid);
  SetOutputState("Kmu3Pi0SelectionOutput", kOValid);
  fEventSelected = false;
  fKmu3Time = 0.0;
  fKmu3TrackID = -1;
  fPi0SelectionOutput.fPi0Momentum.SetXYZM(0., 0., 0., 0.);
  fPi0SelectionOutput.fTime = 0.0;
  fPi0SelectionOutput.fPosition.SetXYZ(0., 0., 0.);
  fPi0SelectionOutput.fKaonMomentum.SetXYZM(0., 0., 0., 0.);
  fPi0SelectionOutput.fGammaMomenta.first.SetXYZM(0., 0., 0., 0.);
  fPi0SelectionOutput.fGammaMomenta.second.SetXYZM(0., 0., 0., 0.);
  fPi0SelectionOutput.fClustersID.first = -1;
  fPi0SelectionOutput.fClustersID.second = -1;

  Bool_t PhysicsTrigger = TriggerConditions::GetInstance()->IsPhysicsTrigger(GetL0Data());
  Bool_t ControlTrigger = TriggerConditions::GetInstance()->IsControlTrigger(GetL0Data());
  Int_t L0TriggerWord = PhysicsTrigger ? GetL0Data()->GetTriggerFlags() : 0;
  Bool_t TriggerOK = (L0TriggerWord & fTriggerMask) || ControlTrigger;

  Int_t BurstID = GetBurstID();
  if(TriggerOK)
    FillHisto("hPhysicsEventsPerBurst", BurstID);

  if(GetWithMC()) {
    Event *evt = GetMCEvent();
    if(evt->GetNKineParts()) {
      FillHisto("mctrue/hZvertex", 0.001 * evt->GetKinePart(0)->GetEndPos().Z());  // [m]
    }
  }

  TRecoLKrEvent *LKrEvent = GetEvent<TRecoLKrEvent>();
  TRecoLAVEvent *LAVEvent = GetEvent<TRecoLAVEvent>();
  TRecoIRCEvent *IRCEvent = GetEvent<TRecoIRCEvent>();
  TRecoSACEvent *SACEvent = GetEvent<TRecoSACEvent>();

  ///////////////////////////////////////
  // Require exactly one good STRAW track

  // Trigger reference time
  Double_t RefTime = GetEventHeader()->GetFineTime() * TdcCalib;

  std::vector<DownstreamTrack> Tracks =
    *GetOutput<std::vector<DownstreamTrack>>("DownstreamTrackBuilder.Output");
  if(TriggerOK)
    FillHisto("hNTracks", Tracks.size());

  if(!Tracks.size())
    return;

  Int_t GoodTrackID = -1;
  Int_t NGoodTracks = 0;
  Double_t GoodTrackTime = 0.0;
  GeometricAcceptance *g = GeometricAcceptance::GetInstance();
  for(UInt_t iTrack = 0; iTrack < Tracks.size(); iTrack++) {

    Double_t TrackTime = 0.0;  // track time
    if(Tracks[iTrack].CHODTimeExists()) {
      TrackTime = Tracks[iTrack].GetCHODTime();
    }
    else if(Tracks[iTrack].NewCHODTimeExists()) {
      TrackTime = Tracks[iTrack].GetNewCHODTime();
    }
    else {
      TrackTime = Tracks[iTrack].GetTrackTime();
    }
    FillHisto("hTrackTime", TrackTime - RefTime);
    //if (Tracks.size()>1 && fabs(TrackTime-RefTime)>8.0) continue;
    if(!Tracks[iTrack].MUV3AssociationExists())
      continue;

    Double_t Ptrack = Tracks[iTrack].GetMomentum();
    Double_t cda = Tracks[iTrack].GetBeamAxisCDA();
    Double_t Zvtx = Tracks[iTrack].GetBeamAxisVertex().Z();

    if(Zvtx < 110000 || Zvtx > 180000)
      continue;
    if(Ptrack < 5000 || Ptrack > 50000)
      continue;
    if(cda > 25.0)
      continue;
    if(Tracks[iTrack].GetCharge() != 1)
      continue;
    if(Tracks[iTrack].GetChi2() > 20.0)
      continue;

    if(!g->InAcceptance(&Tracks[iTrack], NA62::kNewCHOD))
      continue;
    if(!g->InAcceptance(&Tracks[iTrack], NA62::kSpectrometer, 0))
      continue;
    if(!g->InAcceptance(&Tracks[iTrack], NA62::kSpectrometer, 1))
      continue;
    if(!g->InAcceptance(&Tracks[iTrack], NA62::kSpectrometer, 2))
      continue;
    if(!g->InAcceptance(&Tracks[iTrack], NA62::kSpectrometer, 3))
      continue;
    if(!g->InAcceptance(&Tracks[iTrack], NA62::kLKr))
      continue;
    if(!g->InAcceptance(&Tracks[iTrack], NA62::kMUV3))
      continue;

    GoodTrackID = iTrack;
    GoodTrackTime = TrackTime;
    NGoodTracks++;
  }

  if(TriggerOK)
    FillHisto("hNGoodTracks", NGoodTracks);
  if(NGoodTracks != 1)
    return;

  // In-time (wrt good track) track veto to reject K3pi
  Int_t NInTimeTracks = 0;
  for(UInt_t iTrack = 0; iTrack < Tracks.size(); iTrack++) {
    if(iTrack == (UInt_t)GoodTrackID)
      continue;
    Double_t TrackTime = 0.0;  // track time
    if(Tracks[iTrack].CHODTimeExists()) {
      TrackTime = Tracks[iTrack].GetCHODTime();
    }
    else if(Tracks[iTrack].NewCHODTimeExists()) {
      TrackTime = Tracks[iTrack].GetNewCHODTime();
    }
    else {
      TrackTime = Tracks[iTrack].GetTrackTime();
    }
    if(fabs(TrackTime - GoodTrackTime) < 10.)
      NInTimeTracks++;
  }
  if(TriggerOK)
    FillHisto("hNInTimeTracks", NInTimeTracks);
  if(NInTimeTracks > 0)
    return;
  if(!Tracks[GoodTrackID].MUV3AssociationExists())
    return;

  Double_t Ptrack = Tracks[GoodTrackID].GetMomentum();
  Double_t cda = Tracks[GoodTrackID].GetBeamAxisCDA();
  Double_t Zvtx = Tracks[GoodTrackID].GetBeamAxisVertex().Z();

  ////////////////
  // LKr selection

  Int_t N_photon_candidates = 0;
  Double_t Pi0Time = 0.0;
  Double_t x[10], y[10], PhotonEnergy[10];
  Int_t PhotonCandID[10];
  Double_t zlkr = GeometricAcceptance::GetInstance()->GetZLKr();

  for(Int_t i = 0; i < LKrEvent->GetNCandidates(); i++) {
    TRecoLKrCandidate *Lcand = static_cast<TRecoLKrCandidate *>(LKrEvent->GetCandidate(i));
    Double_t LKrTime = Lcand->GetTime();
    Double_t LKrEnergy = Lcand->GetEnergy();
    Double_t dx = Lcand->GetX() - Tracks[GoodTrackID].xAt(zlkr);
    Double_t dy = Lcand->GetY() - Tracks[GoodTrackID].yAt(zlkr);
    Double_t R = sqrt(dx * dx + dy * dy);
    if(fabs(LKrTime - GoodTrackTime) < 6.0 && R > 150.0) {
      FillHisto("hClusterEnergy", 0.001 * LKrEnergy);
    }
    if(LKrEnergy > 2000.0) {
      FillHisto("hClusterTime", LKrTime - GoodTrackTime);
      FillHisto("hClusterDistance", R);
    }
    if(LKrEnergy > 2000.0 && fabs(LKrTime - GoodTrackTime) < 6.0 && R > 150.0
       && N_photon_candidates < 10) {
      Pi0Time += LKrTime;
      x[N_photon_candidates] = Lcand->GetX();
      y[N_photon_candidates] = Lcand->GetY();
      PhotonEnergy[N_photon_candidates] = LKrEnergy;
      PhotonCandID[N_photon_candidates] = i;
      N_photon_candidates++;
    }
  }
  FillHisto("hNPhotonCandidates", N_photon_candidates);
  if(N_photon_candidates != 2)
    return;  // no LKr acceptance check for photons

  Pi0Time *= 0.5;  // averaged over the two photon times

  // Neutral vertex position
  Double_t dist = sqrt((x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1]));
  Double_t ZNVtx = zlkr - dist * sqrt(PhotonEnergy[0] * PhotonEnergy[1]) / MP0;
  Double_t XNVtx = BeamParameters::GetInstance()->GetBeamXatZ(ZNVtx);
  Double_t YNVtx = BeamParameters::GetInstance()->GetBeamYatZ(ZNVtx);

  Double_t dz = Zvtx - ZNVtx;
  FillHisto("hdZChargedNeutralVertex", 1e-3 * dz);
  if(fabs(dz) > 10000)
    return;

  // Photon 4-momenta
  TLorentzVector Photon[2];
  for(Int_t i = 0; i < 2; i++) {
    Double_t norm = sqrt((XNVtx - x[i]) * (XNVtx - x[i]) + (YNVtx - y[i]) * (YNVtx - y[i])
                         + (ZNVtx - zlkr) * (ZNVtx - zlkr));
    Photon[i].SetX(PhotonEnergy[i] * (x[i] - XNVtx) / norm);
    Photon[i].SetY(PhotonEnergy[i] * (y[i] - YNVtx) / norm);
    Photon[i].SetZ(PhotonEnergy[i] * (zlkr - ZNVtx) / norm);
    Photon[i].SetE(PhotonEnergy[i]);
  }
  TLorentzVector Pi0 = Photon[0] + Photon[1];
  // Temporarily filling the pi0 output by hand, but this analyser should be migrated to use the Pi0Selection!
  fPi0SelectionOutput.fPi0Momentum = Pi0;
  fPi0SelectionOutput.fTime = Pi0Time;
  fPi0SelectionOutput.fPosition.SetXYZ(XNVtx, YNVtx, ZNVtx);
  fPi0SelectionOutput.fKaonMomentum.SetVectM(BeamParameters::GetInstance()->GetBeamThreeMomentum(),
                                             MKCH);
  fPi0SelectionOutput.fGammaMomenta.first = Photon[0];
  fPi0SelectionOutput.fGammaMomenta.second = Photon[1];
  fPi0SelectionOutput.fClustersID.first = PhotonCandID[0];
  fPi0SelectionOutput.fClustersID.second = PhotonCandID[1];

  TLorentzVector Kaon;
  Kaon.SetVectM(BeamParameters::GetInstance()->GetBeamThreeMomentum(), MKCH);
  TLorentzVector Muon;
  Muon.SetVectM(Tracks[GoodTrackID].GetMomentumBeforeMagnet(), MMU);
  TLorentzVector Pion;
  Pion.SetVectM(Tracks[GoodTrackID].GetMomentumBeforeMagnet(), MPI);
  Double_t Theta = Kaon.Angle(Tracks[GoodTrackID].GetMomentumBeforeMagnet());

  /////////////////////////
  // LAV veto (with timing)

  LAVMatching *pLAVMatching = *(LAVMatching **)GetOutput("PhotonVetoHandler.LAVMatching");
  pLAVMatching->SetReferenceTime(GoodTrackTime);
  pLAVMatching->SetTimeCuts(3.0, 3.0);
  if(pLAVMatching->LAVHasTimeMatching(LAVEvent))
    return;

  /////////////////////////////////
  // IRC and SAC veto (with timing)

  SAVMatching *pSAVMatching = *(SAVMatching **)GetOutput("PhotonVetoHandler.SAVMatching");
  pSAVMatching->SetReferenceTime(GoodTrackTime);
  pSAVMatching->SetIRCTimeCuts(3.0, 3.0);
  pSAVMatching->SetSACTimeCuts(4.0, 4.0);
  Bool_t SAVmatched = pSAVMatching->SAVHasTimeMatching(IRCEvent, SACEvent);
  if(SAVmatched)
    return;

  /////////////////////////////////

  Double_t eop = Tracks[GoodTrackID].GetLKrEoP();
  Double_t MMiss2_Kmu3 = (Kaon - Muon - Pi0).M2();
  Double_t MMiss2_K2pi = (Kaon - Pion - Pi0).M2();
  Double_t MMiss2_El = (Kaon - Muon).M2();
  Double_t MMiss2_Pi = (Kaon - Pion).M2();
  Double_t Mass2_Kmu3 = (Muon + Pi0).M2();
  Double_t Mass2_K2pi = (Pion + Pi0).M2();
  Double_t Ptot = (Muon + Pi0).P();
  Double_t Pttot = (Muon + Pi0).Perp(Kaon.Vect());

  if(TriggerOK) {
    FillHisto("hPttot", Pttot);
    FillHisto("hPtot", 1e-3 * Ptot);
  }

  if(Ptot < 15000.0 || Ptot > 70000.0)
    return;
  if(Pttot < 40.0 || Pttot > 250.0)
    return;

  if(TriggerOK) {
    FillHisto("hMMiss2_Kmu3", 1.e-6 * MMiss2_Kmu3);  // [GeV^2]
    FillHisto("hPMMiss2_Kmu3", 1e-3 * Ptrack, 1.e-6 * MMiss2_Kmu3);
    FillHisto("hMMiss2_K2pi", 1.e-6 * MMiss2_K2pi);  // [GeV^2]
    FillHisto("hPMMiss2_K2pi", 1e-3 * Ptrack, 1.e-6 * MMiss2_K2pi);
    FillHisto("hMMiss2_El", 1.e-6 * MMiss2_El);  // [GeV^2]
    FillHisto("hPMMiss2_El", 1e-3 * Ptrack, 1.e-6 * MMiss2_El);
    FillHisto("hMMiss2_Pi", 1.e-6 * MMiss2_Pi);  // [GeV^2]
    FillHisto("hPMMiss2_Pi", 1e-3 * Ptrack, 1.e-6 * MMiss2_Pi);
    FillHisto("hMass2_Kmu3", 1.e-6 * Mass2_Kmu3);  // [GeV^2]
    FillHisto("hPMass2_Kmu3", 1e-3 * Ptrack, 1.e-6 * Mass2_Kmu3);
    FillHisto("hMass2_K2pi", 1.e-6 * Mass2_K2pi);  // [GeV^2]
    FillHisto("hPMass2_K2pi", 1e-3 * Ptrack, 1.e-6 * Mass2_K2pi);
    FillHisto("hPTheta", 0.001 * Ptrack, Theta);
  }

  fEventSelected = (fabs(MMiss2_Kmu3) < 1e4);  // 0.01 GeV^2

  if(TriggerOK && fEventSelected) {
    FillHisto("hEOP", eop);
    FillHisto("hZvtx", 0.001 * Zvtx);  // used to compute acceptance
    FillHisto("hCDA", cda);
  }

  if(fEventSelected) {
    fKmu3TrackID = GoodTrackID;
    fKmu3Time = 0.5 * (GoodTrackTime + Pi0Time);
    FillHisto("hKmu3Time", fKmu3Time);
    FillHisto("hKmu3EventsPerBurst", BurstID);

    Int_t nCells = LKrEvent->GetNHits();
    Int_t nClusters = LKrEvent->GetNCandidates();
    FillHisto("LKr/hNLKrCells", nCells);
    FillHisto("LKr/hNLKrClusters", nClusters);

    Double_t totalCellEnergy = 0.;
    for(Int_t i = 0; i < nCells; i++) {
      TRecoLKrHit *hit = static_cast<TRecoLKrHit *>(LKrEvent->GetHit(i));
      Double_t energy = hit->GetEnergy();
      if(energy > 40.0)
        totalCellEnergy += energy;
    }
    FillHisto("LKr/hLKrCellTotalEnergy", 0.001 * totalCellEnergy);
    FillHisto("LKr/hLKrCellClusterTotalEnergy", 0.001 * totalCellEnergy,
              0.001 * LKrEvent->GetEnergyTotal());

    Double_t refTime = TriggerConditions::GetInstance()->IsControlTrigger(GetL0Data())
                         ? GetL0Data()
                             ->GetPrimitive(NA62::Trigger::kL0TriggerSlot, NA62::Trigger::kL0CHOD)
                             .GetFineTime()
                         : GetL0Data()
                             ->GetPrimitive(NA62::Trigger::kL0TriggerSlot, NA62::Trigger::kL0RICH)
                             .GetFineTime();
    refTime *= TdcCalib;
    for(Int_t i = 0; i < nClusters; i++) {
      TRecoLKrCandidate *Lcand = static_cast<TRecoLKrCandidate *>(LKrEvent->GetCandidate(i));
      FillHisto("LKr/hLKrClusterEnergy", 0.001 * Lcand->GetEnergy());
      FillHisto("LKr/hLKrClusterTime", Lcand->GetTime() - refTime);
    }
  }
}

void Kmu3Selection::EndOfJobUser() {
  if(fReadingData) {  // Data mode: save output
    SaveAllPlots();
    return;
  }
  if(!fHZvtx) {  // Histo mode required but no histograms found
    std::cout << user_normal() << "Asked to read my own output but cannot found it" << std::endl;
    return;
  }

  /////////////////////////////////////
  // Histo mode: analyze the histograms

  // Print out acceptance (this is used for revision metrics)
  if(GetWithMC() && fHZtrue) {
    Double_t n = fHZvtx->Integral();
    Double_t N = fHZtrue->Integral(106, 180);  // 105 m < Ztrue < 180 m
    Double_t Acc = n / N;
    Double_t dAcc = sqrt(Acc * (1.0 - Acc) / N);
    std::cout << user_normal() << Form("MC events read: %d\n", (Int_t)fHZtrue->Integral());
    std::cout << user_normal()
              << Form("MC acceptance = %d/%d = %7.5f +- %7.5f\n", (Int_t)n, (Int_t)N, Acc, dAcc);
    std::cout << user_normal() << Form("##CI_DASH::Acceptance.Kmu3=%7.5f+-%7.5f", Acc, dAcc)
              << std::endl;
  }

  BuildPDFReport();
}

void Kmu3Selection::BuildPDFReport() {

  TString OutputPDFFileName = fAnalyzerName + ".pdf";
  gErrorIgnoreLevel = 5000;  // suppress messages generated for each page printed
  gStyle->SetOptStat(11);

  TCanvas *Canvas = new TCanvas("Kmu3Canvas");
  Canvas->Print(Form(OutputPDFFileName + "["), "pdf");  // open file

  Canvas->Divide(2, 2);
  for(Int_t i = 1; i <= 4; i++) {
    Canvas->GetPad(i)->SetLeftMargin(0.1);
    Canvas->GetPad(i)->SetRightMargin(0.03);
    Canvas->GetPad(i)->SetTopMargin(0.06);
    Canvas->GetPad(i)->SetBottomMargin(0.10);
  }
  fHMass->SetLineColor(kBlue);
  fHMass->SetFillColor(kYellow);
  fHEOP->SetLineColor(kBlue);
  fHEOP->SetFillColor(kYellow);
  fHEOP->SetLineColor(kRed);
  fHEOP->SetFillColor(kOrange - 3);
  fHMassVsMomentum->SetMarkerColor(kBlue);

  TLegend *Legend = new TLegend(0.10, 0.75, 0.35, 0.90);
  Legend->SetFillColor(kWhite);
  Legend->AddEntry(fHEOP, "E/P", "pl");
  Legend->AddEntry(fHEOP, "E/P for selected Events", "pl");

  Canvas->cd(1);
  gPad->SetLogy();
  fHEOP->Draw();
  fHEOP->Draw("same");
  Legend->Draw();
  Canvas->cd(2);
  fHMass->Fit("gaus", "", "", -0.005, 0.005);
  fHMass->SetAxisRange(-0.1, 0.1, "X");
  fHMass->Draw();
  Canvas->cd(3);
  fHMassVsMomentum->Draw();
  Canvas->cd(4);
  Int_t MaxNonEmptyBurstID = 0;
  Int_t MaxY = -99;
  for(Int_t i = 0; i < fMaxNBursts; i++) {
    if(fHKmu3EventsPerBurst->GetBinContent(i) > 0) {
      MaxNonEmptyBurstID = i;
      if(fHKmu3EventsPerBurst->GetBinContent(i) > MaxY)
        MaxY = fHKmu3EventsPerBurst->GetBinContent(i);
    }
  }
  TH1F *Kmu3EventsPerBurst = new TH1F("Kmu3EventsPerBurst", "Kmu3 Events per Burst",
                                     MaxNonEmptyBurstID, -0.5, MaxNonEmptyBurstID - 0.5);
  for(Int_t iBin = 1; iBin <= (fHKmu3EventsPerBurst->GetNbinsX()); iBin++) {
    Kmu3EventsPerBurst->SetBinContent(iBin, fHKmu3EventsPerBurst->GetBinContent(iBin));
  }
  Kmu3EventsPerBurst->SetLineColor(kBlue);
  Kmu3EventsPerBurst->SetFillColor(kYellow);
  Kmu3EventsPerBurst->SetAxisRange(0, MaxY + 30, "Y");
  Kmu3EventsPerBurst->GetXaxis()->SetTitle("Burst ID");
  Kmu3EventsPerBurst->Draw();

  Canvas->Print(OutputPDFFileName, "pdf");

  Canvas->Clear();
  Canvas->Divide(1, 2);
  for(Int_t i = 1; i <= 2; i++) {
    Canvas->GetPad(i)->SetLeftMargin(0.04);
    Canvas->GetPad(i)->SetRightMargin(0.01);
    Canvas->GetPad(i)->SetTopMargin(0.06);
    Canvas->GetPad(i)->SetBottomMargin(0.10);
  }
  fHZvtx->SetLineColor(kBlue);
  fHZvtx->SetFillColor(kYellow);

  Canvas->cd(1);
  fHZvtx->Draw();
  Canvas->cd(2);

  /////////////////////////////////////////////////////////////
  // Acceptance of the analyzer as a function of the Z position

  if(fHZtrue) {
    Double_t n = fHZvtx->Integral();
    Double_t N = fHZtrue->Integral(106, 180);  // 105 m < Ztrue < 180 m
    Double_t AcceptanceNew = n / N;
    TString String = Form("Current Acceptance: %5.3f", AcceptanceNew);

    TH1F *fHZtrue2 = new TH1F("Zvertex position for MC true with different bin ranges",
                              "Zvertex position for MC true", 200, 50, 250);
    for(Int_t iBin = 1; iBin <= (fHZtrue->GetNbinsX()); iBin++) {
      fHZtrue2->SetBinContent(fHZtrue2->FindBin(iBin), fHZtrue->GetBinContent(iBin));
    }
    TH1F *fHAcceptanceZvtx =
      new TH1F("AcceptanceatZvtx", "Acceptance at different Z-Positions", 200, 50, 250);
    fHAcceptanceZvtx->Divide(fHZvtx, fHZtrue2, 1., 1., "B");
    fHAcceptanceZvtx->GetXaxis()->SetTitle("Vertex z [m]");

    TLatex text;
    fHAcceptanceZvtx->SetAxisRange(0, 0.3, "Y");
    fHAcceptanceZvtx->Draw();
    text.SetTextSize(0.04);
    text.DrawLatex(105., 0.25, String);
    text.SetNDC(kTRUE);
  }
  Canvas->Print(OutputPDFFileName, "pdf");
  Canvas->Print(Form(OutputPDFFileName + "]"), "pdf");  // close file
  gErrorIgnoreLevel = -1;                               // restore the default

  delete Legend;
  delete Canvas;
  // PrintStatisticsPerBurst();
}

void Kmu3Selection::PrintStatisticsPerBurst() {
  for(Int_t i = 1; i <= fHPhysicsEventsPerBurst->GetNbinsX(); i++) {
    Double_t N = fHPhysicsEventsPerBurst->GetBinContent(i);
    if(!N)
      continue;
    Double_t n = fHKmu3EventsPerBurst->GetBinContent(i);
    Double_t e = n / N;
    Double_t de = sqrt(e * (1.0 - e) / N);
    std::cout << user_standard() << "@@Kmu3 " << i - 1 << " " << n << " " << N << " " << e << " "
              << de << std::endl;
  }
}

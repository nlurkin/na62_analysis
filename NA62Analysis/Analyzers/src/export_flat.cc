#include "export_flat.hh"

#include "Event.hh"
#include "Persistency.hh"
#include "functions.hh"
#include "DownstreamTrack.hh"
#include "SpectrometerTrackVertex.hh"
#include "BeamParameters.hh"
#include "Pi0Selection.hh"
#include "EnergyCluster.hh"
#include "TwoLinesCDA.hh"
#include "Kmu3SelectionNoSpectrometer.hh"

#include <iostream>
#include <stdlib.h>
#include <TH2I.h>
using namespace NA62Analysis;

export_flat::export_flat(Core::BaseAnalysis *ba): Analyzer(ba, "export_flat") {
  RequestTree(new TRecoLKrEvent);
  RequestTree(new TRecoSpectrometerEvent);
  RequestTree(new TRecoRICHEvent);
  RequestTree(new TRecoMUV3Event);
  RequestTree(new TRecoCedarEvent);
  RequestTree(new TRecoGigaTrackerEvent);

  fDSCount = 0;
  fDownscaling = 1000;
}

void export_flat::BranchTrack(TTree* tree, TrackStruct &ts, int it){
  tree->Branch(TString::Format("track%i_exists", it), &(ts.fExists));
  tree->Branch(TString::Format("track%i_has_muv3", it), &(ts.fHasMUV3));
  tree->Branch(TString::Format("track%i_muv3_time", it), &(ts.fMUV3Time));
  tree->Branch(TString::Format("track%i_rich_hypothesis", it), &(ts.fRICHMLH));
  tree->Branch(TString::Format("track%i_rich_radius", it), &(ts.fRICHRadius));
  tree->Branch(TString::Format("track%i_rich_nhits", it), &(ts.fRICHNHits));
  tree->Branch(TString::Format("track%i_rich_center_x", it), &(ts.fRICHCenterX));
  tree->Branch(TString::Format("track%i_rich_center_y", it), &(ts.fRICHCenterY));
  tree->Branch(TString::Format("track%i_direction_x", it), &(ts.fDirectionX));
  tree->Branch(TString::Format("track%i_direction_y", it), &(ts.fDirectionY));
  tree->Branch(TString::Format("track%i_direction_z", it), &(ts.fDirectionZ));
  tree->Branch(TString::Format("track%i_direction_am_x", it), &(ts.fDirectionAfterMagnetX));
  tree->Branch(TString::Format("track%i_direction_am_y", it), &(ts.fDirectionAfterMagnetY));
  tree->Branch(TString::Format("track%i_direction_am_z", it), &(ts.fDirectionAfterMagnetZ));
  tree->Branch(TString::Format("track%i_position_am_x", it), &(ts.fPosAfterMagnetX));
  tree->Branch(TString::Format("track%i_position_am_y", it), &(ts.fPosAfterMagnetY));
  tree->Branch(TString::Format("track%i_momentum_mag", it), &(ts.fMomentumMag));
  tree->Branch(TString::Format("track%i_time", it), &(ts.fTime));
  tree->Branch(TString::Format("track%i_lkr_energy", it), &(ts.fLKrEnergy));
  tree->Branch(TString::Format("track%i_charge", it), &(ts.fCharge));
}

void export_flat::BranchCluster(GammaStruct &ts, int it){
  myTree->Branch(TString::Format("cluster%i_exists", it), &(ts.fExists));
  myTree->Branch(TString::Format("cluster%i_lkr_energy", it), &(ts.fLKrEnergy));
  myTree->Branch(TString::Format("cluster%i_position_x", it), &(ts.fPositionX));
  myTree->Branch(TString::Format("cluster%i_position_y", it), &(ts.fPositionY));
  myTree->Branch(TString::Format("cluster%i_time", it), &(ts.fTime));
}

void export_flat::InitOutput() {
  myTree = new TTree("NA62Flat", "NA62Flat");

  myTree->Branch("run", &fRunNumber);
  myTree->Branch("burst", &fBurstNumber);
  myTree->Branch("event_type", &fEventType);
  myTree->Branch("event_time", &fEventTime);
  myTree->Branch("ReferenceTime", &fReferenceTime);
  myTree->Branch("beam_momentum_mag", &(flat.fBeam.fMomentum));
  myTree->Branch("beam_direction_x", &(flat.fBeam.fDirectionX));
  myTree->Branch("beam_direction_y", &(flat.fBeam.fDirectionY));
  myTree->Branch("beam_direction_z", &(flat.fBeam.fDirectionZ));
  myTree->Branch("beam_position_x", &(flat.fBeam.fPosX));
  myTree->Branch("beam_position_Y", &(flat.fBeam.fPosY));
  myTree->Branch("beam_position_z", &(flat.fBeam.fPosZ));
  myTree->Branch("vtx_x", &(flat.fVertex.fX));
  myTree->Branch("vtx_y", &(flat.fVertex.fY));
  myTree->Branch("vtx_z", &(flat.fVertex.fZ));
  BranchTrack(myTree, flat.track1, 1);
  BranchTrack(myTree, flat.track2, 2);
  BranchTrack(myTree, flat.track3, 3);
  BranchCluster(flat.clus1, 1);
  BranchCluster(flat.clus2, 2);

  if(GetWithMC()){
    myMCTree = new TTree("NA62MCFlat", "NA62MCFlat");
    myMCTree->Branch("run", &fRunNumber);
    myMCTree->Branch("burst", &fBurstNumber);
    myMCTree->Branch("event_type", &fEventType);
    myMCTree->Branch("event_time", &fEventTime);
    myMCTree->Branch("vtx_x", &(mcFlat.fVertex.fX));
    myMCTree->Branch("vtx_y", &(mcFlat.fVertex.fY));
    myMCTree->Branch("vtx_z", &(mcFlat.fVertex.fZ));
    BranchTrack(myMCTree, mcFlat.fKaon, 0);
    BranchTrack(myMCTree, mcFlat.track1, 1);
    BranchTrack(myMCTree, mcFlat.track2, 2);
    BranchTrack(myMCTree, mcFlat.track3, 3);
  }
}

void export_flat::InitHist() {

  BookHisto(new TH1I("muv3tdiff", "muv3tdiff", 100, -10, 10));
  const char *selections[7]  = {"Events", "k3pi", "ke3", "kmu2", "k2pi", "kmu3", "autopass"};
  TH2I *h = new TH2I("sel_matrix", "sel_matrix", 7, -0.5, 6.5, 7, -0.5, 6.5);
  BookHisto(h);
  for (int i =0; i<7; ++i){
    h->GetXaxis()->SetBinLabel(i+1, selections[i]);
    h->GetYaxis()->SetBinLabel(i+1, selections[i]);
  }
}

void export_flat::StartOfRunUser() {
}

void export_flat::StartOfBurstUser() {
}

void export_flat::ProcessSpecialTriggerUser(int , unsigned int ) {
}

void export_flat::Reset() {
  flat.track1.Reset();
  flat.track2.Reset();
  flat.track3.Reset();
  flat.clus1.Reset();
  flat.clus2.Reset();
  flat.fVertex.Reset();
  fKTAGTime = -999.;
}

void export_flat::Process(int) {
  fRunNumber = GetRunID();
  fBurstNumber = GetBurstID();
  fReferenceTime = GetEventHeader()->GetFineTime() * TdcCalib;
  fEventTime = GetEventHeader()->GetTimeStamp();

  std::vector<DownstreamTrack> dsTracks = *GetOutput<std::vector<DownstreamTrack>>("DownstreamTrackBuilder.Output");
  std::vector<SpectrometerTrackVertex> vertices3 = *GetOutput<std::vector<SpectrometerTrackVertex>>("SpectrometerVertexBuilder.Output3");


  Bool_t k3pi_sel = *GetOutput<Bool_t>("K3piSelection.EventSelected");
  Int_t  k3piVertex = *GetOutput<Int_t>("K3piSelection.VertexID");

  Bool_t ke3_sel = *GetOutput<Bool_t>("Ke3Selection.EventSelected");
  Int_t  ke3Track = *GetOutput<Int_t>("Ke3Selection.Ke3TrackID");
  Pi0SelectionOutput ke3Pi0 = *GetOutput<Pi0SelectionOutput>("Ke3Selection.Ke3Pi0SelectionOutput");

  Bool_t kmu2_sel = *GetOutput<Bool_t>("Kmu2Selection.EventSelected");
  Int_t  kmu2Track = *GetOutput<Int_t>("Kmu2Selection.Kmu2TrackID");

  Bool_t kmu3_sel = *GetOutput<Bool_t>("Kmu3Selection.EventSelected");
  Int_t  kmu3Track = *GetOutput<Int_t>("Kmu3Selection.Kmu3TrackID");
  Pi0SelectionOutput kmu3Pi0 = *GetOutput<Pi0SelectionOutput>("Kmu3Selection.Kmu3Pi0SelectionOutput");

  Bool_t k2pi_sel = *GetOutput<Bool_t>("K2piSelection.EventSelected");
  Int_t  k2piTrack = *GetOutput<Int_t>("K2piSelection.K2piTrackID");
  Pi0SelectionOutput k2piPi0 = *GetOutput<Pi0SelectionOutput>("K2piSelection.K2piPi0SelectionOutput");
  TVector3 k2piVtx = *GetOutput<TVector3>("K2piSelection.K2piVertexPosition");

  Int_t autopassVtx = -1;
  Int_t autopassTrack = -1;
  Bool_t autopass_sel = autopassSelection(vertices3, dsTracks, autopassVtx, autopassTrack);

  FillHisto2("sel_matrix", "Events", "Events", 1);
  if(k3pi_sel){
    FillHisto2("sel_matrix", "k3pi", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "k3pi", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "k3pi", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "k3pi", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "k3pi", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "k3pi", "autopass", 1);
  }
  if(ke3_sel){
    FillHisto2("sel_matrix", "ke3", "ke3", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "ke3", "k3pi", 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "ke3", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "ke3", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "ke3", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "ke3", "autopass", 1);
  }
  if(kmu2_sel){
    FillHisto2("sel_matrix", "kmu2", "kmu2", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "kmu2", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "kmu2", "ke3" , 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "kmu2", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "kmu2", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "kmu2", "autopass", 1);
  }
  if(k2pi_sel){
    FillHisto2("sel_matrix", "k2pi", "k2pi", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "k2pi", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "k2pi", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "k2pi", "kmu2", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "k2pi", "kmu3", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "k2pi", "autopass", 1);
  }
  if(kmu3_sel){
    FillHisto2("sel_matrix", "kmu3", "kmu3", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "kmu3", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "kmu3", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "kmu3", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "kmu3", "k2pi", 1);
    if(autopass_sel) FillHisto2("sel_matrix", "kmu3", "autopass", 1);
  }
  if(autopass_sel){
    FillHisto2("sel_matrix", "autopass", "autopass", 1);
    if(k3pi_sel) FillHisto2("sel_matrix", "autopass", "k3pi", 1);
    if(ke3_sel ) FillHisto2("sel_matrix", "autopass", "ke3" , 1);
    if(kmu2_sel) FillHisto2("sel_matrix", "autopass", "kmu2", 1);
    if(k2pi_sel) FillHisto2("sel_matrix", "autopass", "k2pi", 1);
    if(kmu3_sel) FillHisto2("sel_matrix", "autopass", "kmu3", 1);
  }
  Int_t nSel = k3pi_sel + ke3_sel + kmu2_sel + k2pi_sel + kmu3_sel;
  if((nSel==0) && !autopass_sel) return; // Allow at least one positive selection, or autopass

  BeamParameters *beamPar = BeamParameters::GetInstance();
  flat.fBeam.fMomentum = beamPar->GetBeamMomentum();
  TVector3 beam = beamPar->GetBeamThreeMomentum().Unit();
  flat.fBeam.fDirectionX = beam.X();
  flat.fBeam.fDirectionY = beam.Y();
  flat.fBeam.fDirectionZ = beam.Z();
  flat.fBeam.fPosX = beamPar->GetBeamX();
  flat.fBeam.fPosY = beamPar->GetBeamY();
  flat.fBeam.fPosZ = beamPar->GetBeamZ();

  if(autopass_sel) {
    Reset();
    fEventType = 6;

    if(autopassVtx!=-1){
      DownstreamTrack &t1 = dsTracks[vertices3[autopassVtx].GetTrackIndex(0)];
      DownstreamTrack &t2 = dsTracks[vertices3[autopassVtx].GetTrackIndex(1)];
      DownstreamTrack &t3 = dsTracks[vertices3[autopassVtx].GetTrackIndex(2)];

      SpectrometerTrackVertex &vtx = vertices3[k3piVertex];
      float vertexTime = vtx.GetTime();
      flat.fVertex.fX = vtx.GetPosition().X();
      flat.fVertex.fY = vtx.GetPosition().Y();
      flat.fVertex.fZ = vtx.GetPosition().Z();
      FillTrack(t1, flat.track1, vertexTime);
      FillTrack(t2, flat.track2, vertexTime);
      FillTrack(t3, flat.track3, vertexTime);
      TRecoCedarCandidate *ktag = bestKTAGCandidate(vertexTime);
      if(ktag)
        fKTAGTime = ktag->GetTime();
    }
    else{ // Necessarily autopassTrack is set
      DownstreamTrack &t1 = dsTracks[autopassTrack];
      float trackTime = t1.GetMostPreciseTime();

      flat.fVertex.fX = t1.GetBeamAxisVertex().X();
      flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
      flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
      FillTrack(t1, flat.track1, trackTime);
      TRecoCedarCandidate *ktag = bestKTAGCandidate(trackTime);
      if(ktag)
        fKTAGTime = ktag->GetTime();

    }

    // Now check presence of additional clusters
    auto clusters = *GetOutput<std::vector<EnergyCluster>>("EnergyClusterBuilder.Output");
    std::vector<int> goodClusters = additionalClusters(clusters);
    if(goodClusters.size()>=1)
      FillCluster(clusters[goodClusters[0]], flat.clus1);
    if(goodClusters.size()>=2)
      FillCluster(clusters[goodClusters[1]], flat.clus2);

    myTree->Fill();
  }
  if(k3pi_sel) {
    Reset();
    fEventType = 1;
    DownstreamTrack &t1 = dsTracks[vertices3[k3piVertex].GetTrackIndex(0)];
    DownstreamTrack &t2 = dsTracks[vertices3[k3piVertex].GetTrackIndex(1)];
    DownstreamTrack &t3 = dsTracks[vertices3[k3piVertex].GetTrackIndex(2)];

    SpectrometerTrackVertex &vtx = vertices3[k3piVertex];
    float vertexTime = vtx.GetTime();
    flat.fVertex.fX = vtx.GetPosition().X();
    flat.fVertex.fY = vtx.GetPosition().Y();
    flat.fVertex.fZ = vtx.GetPosition().Z();
    FillTrack(t1, flat.track1, vertexTime);
    FillTrack(t2, flat.track2, vertexTime);
    FillTrack(t3, flat.track3, vertexTime);
    TRecoCedarCandidate *ktag = bestKTAGCandidate(vertexTime);
    if(ktag)
      fKTAGTime = ktag->GetTime();

    myTree->Fill();
  }
  if(ke3_sel) {
    Reset();
    fEventType = 2;
    DownstreamTrack &t1 = dsTracks[ke3Track];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = t1.GetBeamAxisVertex().X();
    flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
    flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
    FillTrack(t1, flat.track1, trackTime);

    auto clusters = *GetOutput<std::vector<EnergyCluster>>("EnergyClusterBuilder.Output");
    FillCluster(clusters[ke3Pi0.fClustersID.first], flat.clus1);
    FillCluster(clusters[ke3Pi0.fClustersID.second], flat.clus2);
    TRecoCedarCandidate *ktag = bestKTAGCandidate(trackTime);
    if(ktag)
      fKTAGTime = ktag->GetTime();
    myTree->Fill();
  }
  if(kmu2_sel) {
    Reset();
    fEventType = 3;
    DownstreamTrack &t1 = dsTracks[kmu2Track];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = t1.GetBeamAxisVertex().X();
    flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
    flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
    FillTrack(t1, flat.track1, trackTime);
    TRecoCedarCandidate *ktag = bestKTAGCandidate(trackTime);
    if(ktag)
      fKTAGTime = ktag->GetTime();
    myTree->Fill();
  }
  if(k2pi_sel) {
    Reset();
    fEventType = 4;
    DownstreamTrack &t1 = dsTracks[k2piTrack];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = k2piVtx.X();
    flat.fVertex.fY = k2piVtx.Y();
    flat.fVertex.fZ = k2piVtx.Z();
    FillTrack(t1, flat.track1, trackTime);
    FillCluster(k2piPi0.fEnergyClusters.first, flat.clus1);
    FillCluster(k2piPi0.fEnergyClusters.second, flat.clus2);
    TRecoCedarCandidate *ktag = bestKTAGCandidate(trackTime);
    if(ktag)
      fKTAGTime = ktag->GetTime();
    myTree->Fill();
  }
  if(kmu3_sel) {
    Reset();
    fEventType = 5;
    DownstreamTrack &t1 = dsTracks[kmu3Track];
    float trackTime = t1.GetMostPreciseTime();

    flat.fVertex.fX = t1.GetBeamAxisVertex().X();
    flat.fVertex.fY = t1.GetBeamAxisVertex().Y();
    flat.fVertex.fZ = t1.GetBeamAxisVertex().Z();
    FillTrack(t1, flat.track1, trackTime);
    TRecoCedarCandidate *ktag = bestKTAGCandidate(trackTime);
    if(ktag)
      fKTAGTime = ktag->GetTime();

    auto clusters = *GetOutput<std::vector<EnergyCluster>>("EnergyClusterBuilder.Output");
    FillCluster(clusters[kmu3Pi0.fClustersID.first], flat.clus1);
    FillCluster(clusters[kmu3Pi0.fClustersID.second], flat.clus2);
    myTree->Fill();
  }
  if(GetWithMC()) FillMCTruth();
}

void export_flat::FillTrack(DownstreamTrack &t, TrackStruct &ts, float refTime){
  TVector3 mom = t.GetSpectrometerCandidate()->GetThreeMomentumBeforeMagnet();
  ts.fExists = true;
  ts.fDirectionX = mom.Unit().X();
  ts.fDirectionY = mom.Unit().Y();
  ts.fDirectionZ = mom.Unit().Z();

  mom = t.GetSpectrometerCandidate()->GetThreeMomentumAfterMagnet();
  ts.fDirectionAfterMagnetX = mom.Unit().X();
  ts.fDirectionAfterMagnetY = mom.Unit().Y();
  ts.fDirectionAfterMagnetZ = mom.Unit().Z();

  TVector3 pos = t.GetSpectrometerCandidate()->GetPositionAfterMagnet();
  ts.fPosAfterMagnetX = pos.X();
  ts.fPosAfterMagnetY = pos.Y();

  ts.fMomentumMag = t.GetMomentum();
  ts.fTime = t.GetMostPreciseTime();
  ts.fLKrEnergy = t.GetLKrEnergy();
  ts.fRICHMLH = t.GetRICHMostLikelyHypothesis();
  ts.fRICHNHits = t.GetRICHNumberOfInTimeHits();
  ts.fRICHRadius = t.GetRICHRingRadius();
  ts.fCharge = t.GetCharge();
  TVector2 richCenter = t.GetRICHRingCentrePosition();
  ts.fRICHCenterX = richCenter.X();
  ts.fRICHCenterY = richCenter.Y();
  Double_t dtMin = 9999.;
  Int_t closest = -1;
  for(int i = 0; i < t.GetNMUV3AssociationRecords(); ++i){
    Double_t dt = t.GetMUV3Time(i)-ts.fTime;
    FillHisto("muv3tdiff", dt);
    if (dt < dtMin){
      dtMin = dt;
      closest = i;
    }
  }
  ts.fHasMUV3 = (closest!=-1);
  ts.fMUV3Time = ts.fTime + dtMin;
}

void export_flat::FillCluster(EnergyCluster &c, GammaStruct &ts){
  ts.fExists = true;
  ts.fLKrEnergy = c.GetEnergy();
  ts.fPositionX = c.GetX();
  ts.fPositionY = c.GetY();
  ts.fTime = c.GetTime();
}

TRecoCedarCandidate* export_flat::bestKTAGCandidate(Double_t refTime) {
  TRecoCedarEvent *cedar = GetEvent<TRecoCedarEvent>();
  Int_t iBest = -1;
  Double_t dtBest = 9999.;
  for(Int_t iCand=0; iCand<cedar->GetNCandidates(); ++iCand) {
    Double_t dt = fabs(cedar->GetCandidate(iCand)->GetTime() - refTime);
    if (dt < dtBest && dt < 5){ // At most 5ns from the trigger
      dtBest = dt;
      iBest = iCand;
    }
  }
  if (iBest == -1)
    return nullptr;
  return static_cast<TRecoCedarCandidate*>(cedar->GetCandidate(iBest));
}

Int_t export_flat::bestInTimeVertices(std::vector<SpectrometerTrackVertex> &vtc) {
  TRecoCedarCandidate *cedar = bestKTAGCandidate(fReferenceTime);

  Double_t timeBest = cedar->GetTime();

  std::vector<SpectrometerTrackVertex> ret;
  Int_t iBest = -1;
  Double_t dtBest = 9999.;
  Int_t iv = 0;
  for(auto vtx : vtc) {
    Double_t dt = fabs(vtx.GetTime() - timeBest);
    if(dt < dtBest && dt < 5){ // At most 5ns from the best KTAG
      dtBest = dt;
      iBest = iv;
    }
    ++iv;
  }

  return iBest;
}

std::vector<int> export_flat::goodDSTracks(std::vector<DownstreamTrack> &ds) {

  std::vector<int> goodTracks;
  if(ds.size() > 10)
    return goodTracks;

  Int_t trackID = 0;
  for(auto &track : ds) {
    if(!track.CHODTimeExists() && !track.NewCHODAssociationExists())
      continue;

    Double_t TrackTime =
      (track.CHODTimeExists()) ? track.GetCHODTime() : track.GetNewCHODTime();
    TrackTime -= fReferenceTime;  // with respect to the trigger time
    if(fabs(TrackTime) > 10.0)
      continue;  // out of time with the trigger

    if(track.GetNChambers() != 4)
      continue;
    if(track.GetChi2() > 20.0)
      continue;

    goodTracks.push_back(trackID++);
  }

  return goodTracks;
}

Bool_t export_flat::autopassSelection(std::vector<SpectrometerTrackVertex> &vertex3, std::vector<DownstreamTrack> &dsTracks, Int_t &goodvtx, Int_t &goodtrack) {
  if(++fDSCount<fDownscaling) return false;

  fDSCount = 0;

  goodvtx = bestInTimeVertices(vertex3);
  if(goodvtx!=-1) return true;

  std::vector<int> goodTracks = goodDSTracks(dsTracks);
  if(goodTracks.size()==1) {
    goodtrack = goodTracks[0];
    return true;
  }

  return false;
}

std::vector<int> export_flat::additionalClusters(std::vector<EnergyCluster> &clusters) {
  std::vector<int> goodClusters;
  Int_t iClus = 0;
  for(const auto &clus: clusters) {
    if((fabs(clus.GetTime() - fReferenceTime) > 20))
      continue;
    if((clus.GetDDeadCell() < 20.))
      continue;
    if((!clus.GetIsElectromagnetic()))
      continue;
    if((clus.GetEnergy() < 1000.))
      continue;
    if(clus.SpectrometerAssociationExists())
      continue;
    if(clus.GetHasCHODAssociation())
      continue;
    if(!clus.GetIsIsolated())
      continue;
    goodClusters.push_back(iClus++);
  }

  return goodClusters;
}

void export_flat::FillMCTruth() {
  Event *mcevt = GetMCEvent();
  mcFlat.fKaon.Reset();
  mcFlat.track1.Reset();
  mcFlat.track2.Reset();
  mcFlat.track3.Reset();

  if(mcevt == nullptr)
    return;
  EventBoundary *pOriginalEventBoundary = static_cast<EventBoundary *>(mcevt->GetEventBoundary(0));
  if(pOriginalEventBoundary->GetNKineParts() == 0) {
    std::cout << user_normal() << "WARNING: Empty MC event!" << std::endl;
    return;
  }
  int iTrack = 0;
  for(int ikine = pOriginalEventBoundary->GetFirstKinePartIndex();
      ikine < pOriginalEventBoundary->GetLastKinePartIndex() + 1; ikine++) {
    KinePart *kp = static_cast<KinePart *>(mcevt->GetKineParts()->At(ikine));
    if(ikine == pOriginalEventBoundary->GetFirstKinePartIndex() && kp->GetParentIndex() == -1) {
      FillMCTrack(kp, mcFlat.fKaon);
      TLorentzVector endpos = kp->GetEndPos();
      mcFlat.fVertex.fX = endpos.X();
      mcFlat.fVertex.fY = endpos.Y();
      mcFlat.fVertex.fZ = endpos.Z();
      continue;
    }
    if(kp->GetParentIndex() != 0 || !kp->GetProdProcessName().Contains("Decay"))
      continue;  // KinePart not originating from the beam particle

    if(iTrack==0)
      FillMCTrack(kp, mcFlat.track1);
    else if(iTrack==1)
      FillMCTrack(kp, mcFlat.track2);
    else if(iTrack==2)
      FillMCTrack(kp, mcFlat.track3);

    ++iTrack;
  }

  myMCTree->Fill();
}

void export_flat::FillMCTrack(KinePart *t, TrackStruct &ts){
  TVector3 mom = t->GetInitialMomentum();
  ts.fExists = true;
  ts.fDirectionX = mom.Unit().X();
  ts.fDirectionY = mom.Unit().Y();
  ts.fDirectionZ = mom.Unit().Z();

  ts.fMomentumMag = mom.Mag();
  ts.fTime = t->GetProdPos().T();
  ts.fCharge = t->GetCharge();
}

void export_flat::PostProcess() {
}

void export_flat::EndOfBurstUser() {
}

void export_flat::EndOfRunUser() {
}

void export_flat::EndOfJobUser() {
  myTree->Write();
  if(GetWithMC()) myMCTree->Write();
  SaveAllPlots();
}

void export_flat::DrawPlot() {
}

export_flat::~export_flat() {
}

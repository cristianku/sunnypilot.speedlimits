# SYNTHETIC test schema (tests/review/test_drive_review.py), not the comma's: only the services and fields that
# nnslr_tools.drive_extract reads, with the names and types of the runtime branch (openpilot/cereal log.capnp and
# custom.capnp of cristianku/sunnypilot speed-limits). Field numbers and the other services differ: a real log cannot
# be read with it, and the logs written with it are made up by the tests.
@0xe8349cd3debf03c8;

using Car = import "car.capnp";

struct InitData {
  version @0 :Text;
  gitBranch @1 :Text;
  gitCommit @2 :Text;
  deviceType @3 :DeviceType;
  enum DeviceType {
    unknown @0;
    tici @1;
    tizi @2;
    mici @3;
  }
}

struct SelfdriveState {
  state @0 :OpenpilotState;
  enabled @1 :Bool;
  active @2 :Bool;
  alertType @3 :Text;
  enum OpenpilotState {
    disabled @0;
    preEnabled @1;
    enabled @2;
    softDisabling @3;
    overriding @4;
  }
}

struct OnroadEvent {
  name @0 :EventName;
  enable @1 :Bool;
  noEntry @2 :Bool;
  warning @3 :Bool;
  userDisable @4 :Bool;
  softDisable @5 :Bool;
  immediateDisable @6 :Bool;
  preEnable @7 :Bool;
  permanent @8 :Bool;
  overrideLateral @9 :Bool;
  overrideLongitudinal @10 :Bool;
  enum EventName {
    canError @0;
    buttonCancel @1;
    pedalPressed @2;
    steerOverride @3;
    gasPressedOverride @4;
  }
}

struct LongitudinalPlan {
  aTarget @0 :Float32;
}

struct EncodeIndex {
  frameId @0 :UInt32;
  segmentNum @1 :Int32;
  segmentId @2 :UInt32;
  timestampSof @3 :UInt64;
  timestampEof @4 :UInt64;
}

struct GpsLocationData {
  latitude @0 :Float64;
  longitude @1 :Float64;
  hasFix @2 :Bool;
  unixTimestampMillis @3 :Int64;
}

struct SelfdriveStateSP {
  mads @0 :Mads;
  intelligentCruiseButtonManagement @1 :Icbm;
  struct Mads {
    state @0 :Text;
    active @1 :Bool;
  }
  struct Icbm {
    state @0 :Text;
    sendButton @1 :SendButton;
    enum SendButton {
      none @0;
      increase @1;
      decrease @2;
    }
  }
}

struct LongitudinalPlanSP {
  longitudinalPlanSource @0 :Text;
  speedLimit @1 :SpeedLimit;
  vTarget @2 :Float32;
  struct SpeedLimit {
    resolver @0 :Resolver;
    assist @1 :Assist;
    struct Resolver {
      speedLimit @0 :Float32;
      source @1 :Source;
      speedLimitFinal @2 :Float32;
    }
    struct Assist {
      state @0 :AssistState;
      active @1 :Bool;
    }
    enum Source {
      none @0;
      car @1;
      map @2;
      vision @3;
    }
    enum AssistState {
      disabled @0;
      inactive @1;
      preActive @2;
      pending @3;
      adapting @4;
      active @5;
    }
  }
}

struct LiveMapDataSP {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedLimitAheadValid @2 :Bool;
  speedLimitAhead @3 :Float32;
  speedLimitAheadDistance @4 :Float32;
  roadName @5 :Text;
}

struct SpeedVisionSP {
  speedLimit @0 :Float32;
  modelName @1 :Text;
  signRead @2 :Bool;
  roadType @3 :Text;
}

struct Event {
  logMonoTime @0 :UInt64;
  valid @1 :Bool;
  union {
    initData @2 :InitData;
    carState @3 :Car.CarState;
    carControl @4 :Car.CarControl;
    carParams @5 :Car.CarParams;
    selfdriveState @6 :SelfdriveState;
    onroadEvents @7 :List(OnroadEvent);
    longitudinalPlan @8 :LongitudinalPlan;
    roadEncodeIdx @9 :EncodeIndex;
    gpsLocationExternal @10 :GpsLocationData;
    selfdriveStateSP @11 :SelfdriveStateSP;
    longitudinalPlanSP @12 :LongitudinalPlanSP;
    liveMapDataSP @13 :LiveMapDataSP;
    speedVisionSP @14 :SpeedVisionSP;
  }
}

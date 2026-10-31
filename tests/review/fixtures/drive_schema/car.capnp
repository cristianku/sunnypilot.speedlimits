# SYNTHETIC test schema (tests/review/test_drive_review.py), not opendbc's car.capnp: only the fields of carState,
# carControl and carParams that nnslr_tools.drive_extract reads, with their names and types.
@0xcf844b6d7fdd695b;

struct CarState {
  vEgo @0 :Float32;
  gasPressed @1 :Bool;
  brakePressed @2 :Bool;
  steeringPressed @3 :Bool;
  cruiseState @4 :CruiseState;
  vCruise @5 :Float32;
  buttonEvents @6 :List(ButtonEvent);
  struct CruiseState {
    enabled @0 :Bool;
    available @1 :Bool;
    speed @2 :Float32;
  }
  struct ButtonEvent {
    pressed @0 :Bool;
    type @1 :Type;
    enum Type {
      unknown @0;
      leftBlinker @1;
      rightBlinker @2;
      accelCruise @3;
      decelCruise @4;
      cancel @5;
      mainCruise @6;
      setCruise @7;
      resumeCruise @8;
    }
  }
}

struct CarControl {
  latActive @0 :Bool;
  longActive @1 :Bool;
}

struct CarParams {
  brand @0 :Text;
  carFingerprint @1 :Text;
  dashcamOnly @2 :Bool;
  openpilotLongitudinalControl @3 :Bool;
}

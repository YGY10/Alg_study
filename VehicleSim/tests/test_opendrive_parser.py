from pathlib import Path

import pytest

from sim2d.map import (
    OpenDriveArcGeometry,
    OpenDriveLaneSide,
    OpenDriveLineGeometry,
    parse_opendrive_file,
    parse_opendrive_string,
)

SIMPLE_XODR = """
<OpenDRIVE>
    <header
        revMajor="1"
        revMinor="7"
        name="unit_test"
        version="1.00"
        date=""
        north="0"
        south="0"
        east="0"
        west="0"
    />

    <road
        name="Main Road"
        length="20.0"
        id="1"
        junction="-1"
    >
        <planView>
            <geometry
                s="0.0"
                x="0.0"
                y="0.0"
                hdg="0.0"
                length="10.0"
            >
                <line/>
            </geometry>

            <geometry
                s="10.0"
                x="10.0"
                y="0.0"
                hdg="0.0"
                length="10.0"
            >
                <arc curvature="0.1"/>
            </geometry>
        </planView>

        <lanes>
            <laneSection s="0.0">
                <left>
                    <lane
                        id="1"
                        type="driving"
                        level="false"
                    >
                        <link>
                            <successor id="1"/>
                        </link>

                        <width
                            sOffset="0.0"
                            a="3.5"
                            b="0.0"
                            c="0.0"
                            d="0.0"
                        />
                    </lane>
                </left>

                <center>
                    <lane
                        id="0"
                        type="none"
                        level="false"
                    />
                </center>

                <right>
                    <lane
                        id="-1"
                        type="driving"
                        level="false"
                    >
                        <link>
                            <predecessor id="-1"/>
                        </link>

                        <width
                            sOffset="0.0"
                            a="3.5"
                            b="0.0"
                            c="0.0"
                            d="0.0"
                        />
                    </lane>
                </right>
            </laneSection>
        </lanes>
    </road>
</OpenDRIVE>
"""


def test_parse_simple_opendrive() -> None:
    roads = parse_opendrive_string(SIMPLE_XODR)

    assert len(roads) == 1

    road = roads[0]

    assert road.road_id == "1"
    assert road.name == "Main Road"
    assert road.length == pytest.approx(20.0)

    assert road.junction_id is None

    assert len(road.geometries) == 2

    assert isinstance(
        road.geometries[0],
        OpenDriveLineGeometry,
    )

    assert isinstance(
        road.geometries[1],
        OpenDriveArcGeometry,
    )

    assert road.geometries[1].curvature == pytest.approx(0.1)


def test_parse_lane_section() -> None:
    road = parse_opendrive_string(SIMPLE_XODR)[0]

    assert len(road.lane_sections) == 1

    section = road.lane_sections[0]

    left_lane = section.get_lane(1)

    center_lane = section.get_lane(0)

    right_lane = section.get_lane(-1)

    assert left_lane.side is OpenDriveLaneSide.LEFT

    assert center_lane.side is OpenDriveLaneSide.CENTER

    assert right_lane.side is OpenDriveLaneSide.RIGHT

    assert left_lane.successor_id == 1
    assert right_lane.predecessor_id == -1

    assert left_lane.width_at(5.0) == pytest.approx(3.5)


def test_parse_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test_map.xodr"

    path.write_text(
        SIMPLE_XODR,
        encoding="utf-8",
    )

    roads = parse_opendrive_file(path)

    assert len(roads) == 1

    assert roads[0].road_id == "1"


def test_invalid_root_raises() -> None:
    xml = """
    <Map>
        <road
            id="1"
            length="10.0"
        />
    </Map>
    """

    with pytest.raises(
        ValueError,
        match="root element",
    ):
        parse_opendrive_string(xml)


def test_missing_plan_view_raises() -> None:
    xml = """
    <OpenDRIVE>
        <road
            id="1"
            length="10.0"
            junction="-1"
        >
            <lanes>
                <laneSection s="0.0"/>
            </lanes>
        </road>
    </OpenDRIVE>
    """

    with pytest.raises(
        ValueError,
        match="planView",
    ):
        parse_opendrive_string(xml)


def test_unsupported_spiral_raises() -> None:
    xml = """
    <OpenDRIVE>
        <road
            id="1"
            length="10.0"
            junction="-1"
        >
            <planView>
                <geometry
                    s="0.0"
                    x="0.0"
                    y="0.0"
                    hdg="0.0"
                    length="10.0"
                >
                    <spiral
                        curvStart="0.0"
                        curvEnd="0.1"
                    />
                </geometry>
            </planView>

            <lanes>
                <laneSection s="0.0"/>
            </lanes>
        </road>
    </OpenDRIVE>
    """

    with pytest.raises(
        NotImplementedError,
        match="spiral",
    ):
        parse_opendrive_string(xml)


def test_missing_required_attribute_raises() -> None:
    xml = """
    <OpenDRIVE>
        <road
            id="1"
            junction="-1"
        >
            <planView>
                <geometry
                    s="0.0"
                    x="0.0"
                    y="0.0"
                    hdg="0.0"
                    length="10.0"
                >
                    <line/>
                </geometry>
            </planView>

            <lanes>
                <laneSection s="0.0"/>
            </lanes>
        </road>
    </OpenDRIVE>
    """

    with pytest.raises(
        ValueError,
        match="length",
    ):
        parse_opendrive_string(xml)


def test_duplicate_road_ids_raise() -> None:
    road_xml = """
        <road
            id="1"
            length="10.0"
            junction="-1"
        >
            <planView>
                <geometry
                    s="0.0"
                    x="0.0"
                    y="0.0"
                    hdg="0.0"
                    length="10.0"
                >
                    <line/>
                </geometry>
            </planView>

            <lanes>
                <laneSection s="0.0"/>
            </lanes>
        </road>
    """

    xml = "<OpenDRIVE>" + road_xml + road_xml + "</OpenDRIVE>"

    with pytest.raises(
        ValueError,
        match="duplicate road IDs",
    ):
        parse_opendrive_string(xml)


def test_invalid_xml_raises() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid OpenDRIVE XML",
    ):
        parse_opendrive_string("<OpenDRIVE>")


def test_missing_file_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        parse_opendrive_file(tmp_path / "missing.xodr")

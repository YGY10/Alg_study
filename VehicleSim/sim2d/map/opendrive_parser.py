from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from sim2d.map.opendrive_types import (
    OpenDriveArcGeometry,
    OpenDriveGeometry,
    OpenDriveLane,
    OpenDriveLaneSection,
    OpenDriveLaneSide,
    OpenDriveLaneWidth,
    OpenDriveContactPoint,
    OpenDriveElementType,
    OpenDriveJunction,
    OpenDriveJunctionConnection,
    OpenDriveJunctionLaneLink,
    OpenDriveLineGeometry,
    OpenDriveMap,
    OpenDriveRoad,
    OpenDriveRoadLink,
)


def _required_attribute(
    element: ElementTree.Element,
    name: str,
) -> str:
    """
    读取必需 XML 属性。

    属性不存在时抛出带元素名称的 ValueError。
    """
    value = element.get(name)

    if value is None:
        raise ValueError(f"<{element.tag}> is missing required " f"attribute {name!r}")

    return value


def _required_float(
    element: ElementTree.Element,
    name: str,
) -> float:
    """
    读取必需浮点属性。
    """
    text = _required_attribute(
        element,
        name,
    )

    try:
        return float(text)
    except ValueError as error:
        raise ValueError(
            f"<{element.tag}> attribute {name!r} " f"must be a float, got {text!r}"
        ) from error


def _required_int(
    element: ElementTree.Element,
    name: str,
) -> int:
    """
    读取必需整数属性。
    """
    text = _required_attribute(
        element,
        name,
    )

    try:
        return int(text)
    except ValueError as error:
        raise ValueError(
            f"<{element.tag}> attribute {name!r} " f"must be an integer, got {text!r}"
        ) from error


def _optional_int(
    element: ElementTree.Element,
    name: str,
) -> int | None:
    """
    读取可选整数属性。
    """
    text = element.get(name)

    if text is None:
        return None

    try:
        return int(text)
    except ValueError as error:
        raise ValueError(
            f"<{element.tag}> attribute {name!r} " f"must be an integer, got {text!r}"
        ) from error


def _parse_bool(
    value: str | None,
    *,
    default: bool = False,
) -> bool:
    """
    解析 OpenDRIVE 布尔属性。
    """
    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {
        "true",
        "1",
    }:
        return True

    if normalized in {
        "false",
        "0",
    }:
        return False

    raise ValueError(f"Invalid boolean value: {value!r}")


def parse_geometry_element(
    geometry_element: ElementTree.Element,
) -> OpenDriveGeometry:
    """
    解析一个 <geometry>。

    当前支持：

        <line/>
        <arc curvature="..."/>

    当前不支持：

        spiral
        poly3
        paramPoly3
    """
    geometry_s = _required_float(
        geometry_element,
        "s",
    )

    x = _required_float(
        geometry_element,
        "x",
    )

    y = _required_float(
        geometry_element,
        "y",
    )

    heading = _required_float(
        geometry_element,
        "hdg",
    )

    length = _required_float(
        geometry_element,
        "length",
    )

    geometry_children = [
        child
        for child in geometry_element
        if child.tag
        in {
            "line",
            "arc",
            "spiral",
            "poly3",
            "paramPoly3",
        }
    ]

    if len(geometry_children) != 1:
        raise ValueError(
            "<geometry> must contain exactly one " "supported geometry child"
        )

    geometry_child = geometry_children[0]

    if geometry_child.tag == "line":
        return OpenDriveLineGeometry(
            s=geometry_s,
            x=x,
            y=y,
            heading=heading,
            length=length,
        )

    if geometry_child.tag == "arc":
        curvature = _required_float(
            geometry_child,
            "curvature",
        )

        return OpenDriveArcGeometry(
            s=geometry_s,
            x=x,
            y=y,
            heading=heading,
            length=length,
            curvature=curvature,
        )

    raise NotImplementedError(
        "Unsupported OpenDRIVE geometry type: " f"{geometry_child.tag!r}"
    )


def parse_lane_width_element(
    width_element: ElementTree.Element,
) -> OpenDriveLaneWidth:
    """
    解析一个 <width> 三次多项式。
    """
    return OpenDriveLaneWidth(
        s_offset=_required_float(
            width_element,
            "sOffset",
        ),
        a=_required_float(
            width_element,
            "a",
        ),
        b=_required_float(
            width_element,
            "b",
        ),
        c=_required_float(
            width_element,
            "c",
        ),
        d=_required_float(
            width_element,
            "d",
        ),
    )


def _parse_lane_link_id(
    lane_element: ElementTree.Element,
    link_type: str,
) -> int | None:
    """
    读取 lane/link 下的 predecessor 或 successor ID。
    """
    link_element = lane_element.find("link")

    if link_element is None:
        return None

    target_element = link_element.find(link_type)

    if target_element is None:
        return None

    return _required_int(
        target_element,
        "id",
    )


def parse_lane_element(
    lane_element: ElementTree.Element,
    side: OpenDriveLaneSide,
) -> OpenDriveLane:
    """
    解析一个 <lane>。
    """
    widths = tuple(
        sorted(
            (
                parse_lane_width_element(width_element)
                for width_element in lane_element.findall("width")
            ),
            key=lambda width: width.s_offset,
        )
    )

    lane_type = _required_attribute(
        lane_element,
        "type",
    )

    return OpenDriveLane(
        lane_id=_required_int(
            lane_element,
            "id",
        ),
        side=side,
        lane_type=lane_type,
        widths=widths,
        predecessor_id=_parse_lane_link_id(
            lane_element,
            "predecessor",
        ),
        successor_id=_parse_lane_link_id(
            lane_element,
            "successor",
        ),
        level=_parse_bool(
            lane_element.get("level"),
            default=False,
        ),
    )


def _parse_lane_side(
    lane_section_element: ElementTree.Element,
    *,
    tag: str,
    side: OpenDriveLaneSide,
) -> list[OpenDriveLane]:
    """
    解析 laneSection 中某一侧车道。
    """
    side_element = lane_section_element.find(tag)

    if side_element is None:
        return []

    return [
        parse_lane_element(
            lane_element,
            side,
        )
        for lane_element in side_element.findall("lane")
    ]


def parse_lane_section_element(
    lane_section_element: ElementTree.Element,
) -> OpenDriveLaneSection:
    """
    解析一个 <laneSection>。
    """
    lanes: list[OpenDriveLane] = []

    lanes.extend(
        _parse_lane_side(
            lane_section_element,
            tag="left",
            side=OpenDriveLaneSide.LEFT,
        )
    )

    lanes.extend(
        _parse_lane_side(
            lane_section_element,
            tag="center",
            side=OpenDriveLaneSide.CENTER,
        )
    )

    lanes.extend(
        _parse_lane_side(
            lane_section_element,
            tag="right",
            side=OpenDriveLaneSide.RIGHT,
        )
    )

    return OpenDriveLaneSection(
        s=_required_float(
            lane_section_element,
            "s",
        ),
        lanes=tuple(lanes),
    )


def _parse_contact_point(
    value: str,
) -> OpenDriveContactPoint:
    normalized = value.strip().lower()

    try:
        return OpenDriveContactPoint(normalized)
    except ValueError as error:
        raise ValueError(f"Invalid OpenDRIVE contactPoint: {value!r}") from error


def parse_road_link_element(
    link_element: ElementTree.Element,
) -> OpenDriveRoadLink:
    """
    解析 road/link 下的 predecessor 或 successor。
    """
    element_type_text = (
        _required_attribute(
            link_element,
            "elementType",
        )
        .strip()
        .lower()
    )

    try:
        element_type = OpenDriveElementType(element_type_text)
    except ValueError as error:
        raise ValueError(
            "Unsupported road link elementType: " f"{element_type_text!r}"
        ) from error

    element_id = _required_attribute(
        link_element,
        "elementId",
    )

    contact_point_text = link_element.get("contactPoint")

    contact_point: OpenDriveContactPoint | None

    if contact_point_text is None:
        contact_point = None
    else:
        contact_point = _parse_contact_point(contact_point_text)

    return OpenDriveRoadLink(
        element_type=element_type,
        element_id=element_id,
        contact_point=contact_point,
    )


def _parse_optional_road_link(
    road_element: ElementTree.Element,
    relation: str,
) -> OpenDriveRoadLink | None:
    road_link_element = road_element.find("link")

    if road_link_element is None:
        return None

    relation_element = road_link_element.find(relation)

    if relation_element is None:
        return None

    return parse_road_link_element(relation_element)


def parse_junction_lane_link_element(
    lane_link_element: ElementTree.Element,
) -> OpenDriveJunctionLaneLink:
    return OpenDriveJunctionLaneLink(
        from_lane_id=_required_int(
            lane_link_element,
            "from",
        ),
        to_lane_id=_required_int(
            lane_link_element,
            "to",
        ),
    )


def parse_junction_connection_element(
    connection_element: ElementTree.Element,
) -> OpenDriveJunctionConnection:
    lane_links = tuple(
        parse_junction_lane_link_element(lane_link_element)
        for lane_link_element in connection_element.findall("laneLink")
    )

    return OpenDriveJunctionConnection(
        connection_id=_required_attribute(
            connection_element,
            "id",
        ),
        incoming_road_id=_required_attribute(
            connection_element,
            "incomingRoad",
        ),
        connecting_road_id=_required_attribute(
            connection_element,
            "connectingRoad",
        ),
        contact_point=_parse_contact_point(
            _required_attribute(
                connection_element,
                "contactPoint",
            )
        ),
        lane_links=lane_links,
    )


def parse_junction_element(
    junction_element: ElementTree.Element,
) -> OpenDriveJunction:
    return OpenDriveJunction(
        junction_id=_required_attribute(
            junction_element,
            "id",
        ),
        name=junction_element.get("name"),
        connections=tuple(
            parse_junction_connection_element(connection_element)
            for connection_element in junction_element.findall("connection")
        ),
    )


def parse_road_element(
    road_element: ElementTree.Element,
) -> OpenDriveRoad:
    """
    解析一个 <road>。
    """
    road_id = _required_attribute(
        road_element,
        "id",
    )

    road_length = _required_float(
        road_element,
        "length",
    )

    road_name = road_element.get("name")

    raw_junction_id = road_element.get("junction")

    junction_id: str | None

    if raw_junction_id in {
        None,
        "",
        "-1",
    }:
        junction_id = None
    else:
        junction_id = raw_junction_id

    plan_view_element = road_element.find("planView")

    if plan_view_element is None:
        raise ValueError(f"Road {road_id!r} does not contain " "<planView>")

    geometries = tuple(
        parse_geometry_element(geometry_element)
        for geometry_element in plan_view_element.findall("geometry")
    )

    lanes_element = road_element.find("lanes")

    if lanes_element is None:
        raise ValueError(f"Road {road_id!r} does not contain " "<lanes>")

    lane_sections = tuple(
        parse_lane_section_element(lane_section_element)
        for lane_section_element in lanes_element.findall("laneSection")
    )

    return OpenDriveRoad(
        road_id=road_id,
        name=road_name,
        length=road_length,
        geometries=geometries,
        lane_sections=lane_sections,
        junction_id=junction_id,
        predecessor=_parse_optional_road_link(
            road_element,
            "predecessor",
        ),
        successor=_parse_optional_road_link(
            road_element,
            "successor",
        ),
    )


def parse_opendrive_map_root(
    root: ElementTree.Element,
) -> OpenDriveMap:
    """
    从 OpenDRIVE XML 根元素解析完整文档。
    """
    if root.tag != "OpenDRIVE":
        raise ValueError(
            "OpenDRIVE root element must be " f"<OpenDRIVE>, got <{root.tag}>"
        )

    roads = tuple(
        parse_road_element(road_element) for road_element in root.findall("road")
    )

    if not roads:
        raise ValueError("OpenDRIVE document does not contain " "any <road> elements")

    junctions = tuple(
        parse_junction_element(junction_element)
        for junction_element in root.findall("junction")
    )

    return OpenDriveMap(
        roads=roads,
        junctions=junctions,
    )


def parse_opendrive_root(
    root: ElementTree.Element,
) -> tuple[OpenDriveRoad, ...]:
    """
    兼容旧接口：仅返回 roads。
    """
    return parse_opendrive_map_root(root).roads


def parse_opendrive_map_string(
    xml_text: str,
) -> OpenDriveMap:
    """
    从 XML 字符串解析完整 OpenDRIVE 文档。
    """
    if not xml_text.strip():
        raise ValueError("xml_text cannot be empty")

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as error:
        raise ValueError("Invalid OpenDRIVE XML") from error

    return parse_opendrive_map_root(root)


def parse_opendrive_string(
    xml_text: str,
) -> tuple[OpenDriveRoad, ...]:
    """
    兼容旧接口：从 XML 字符串仅返回 roads。
    """
    return parse_opendrive_map_string(xml_text).roads


def parse_opendrive_map_file(
    path: str | Path,
) -> OpenDriveMap:
    """
    从 .xodr 文件解析完整 OpenDRIVE 文档。
    """
    file_path = Path(path).expanduser()

    if not file_path.exists():
        raise FileNotFoundError("OpenDRIVE file does not exist: " f"{file_path}")

    if not file_path.is_file():
        raise ValueError("OpenDRIVE path is not a file: " f"{file_path}")

    try:
        tree = ElementTree.parse(file_path)
    except ElementTree.ParseError as error:
        raise ValueError("Invalid OpenDRIVE XML file: " f"{file_path}") from error

    parsed_map = parse_opendrive_map_root(tree.getroot())

    return OpenDriveMap(
        roads=parsed_map.roads,
        junctions=parsed_map.junctions,
        source_name=file_path.name,
    )


def parse_opendrive_file(
    path: str | Path,
) -> tuple[OpenDriveRoad, ...]:
    """
    兼容旧接口：从 .xodr 文件仅返回 roads。
    """
    return parse_opendrive_map_file(path).roads

"""
Command-line interface for mesh calculator.
"""
import click
import os
from pathlib import Path

import structlog

from ..logging_config import setup_logging
from ..utils.perf import PerfTimer
from ..data.loaders import load_config
from ..core.config import OutputPaths
from ..data.sites import load_sites, snap_sites_to_roads
from ..data.cache import LOSCache
from ..data.exporters import (
    export_towers_geojson, export_coverage_geojson,
    export_visibility_edges_geojson, generate_report,
    export_gap_repair_hexes_geojson,
)
from ..core.elevation import ElevationProvider
from ..core.grid import load_boundary, load_roads, generate_road_grid
from ..network.graph import MeshSurface
from ..network.routing import build_routing_graph
from ..optimization.hierarchical import connect_sites_by_priority

logger = structlog.get_logger(__name__)


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='Path to YAML configuration file')
@click.option('--output', type=click.Path(), default='output',
              help='Output directory path')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', is_flag=True, help='Suppress info-level logging')
def main(config: str, output: str, verbose: bool, quiet: bool):
    """
    Mesh Network Tower Placement Optimizer

    Connects user-specified sites (cities) via mesh network nodes placed along roads
    with hierarchical priority levels.
    """
    setup_logging(verbose=verbose, quiet=quiet)

    logger.info("Mesh Network Tower Placement Optimizer")

    # Create output directory
    os.makedirs(output, exist_ok=True)

    # Load configuration
    logger.info("[1/9] Loading configuration")
    with PerfTimer("load_configuration"):
        cfg = load_config(config)
    logger.info("Configuration loaded",
                h3_resolution=cfg.parameters.h3_resolution,
                max_nodes_per_route=cfg.parameters.max_towers_per_route)

    # Load input data
    logger.info("[2/9] Loading input data")
    with PerfTimer("load_input_data"):
        logger.info("Loading boundary")
        boundary = load_boundary(cfg.inputs.boundary)
        logger.info("Boundary loaded", area_sq_deg=round(boundary.area, 4))

        logger.info("Loading roads")
        roads_gdf = load_roads(cfg.inputs.roads)
        logger.info("Roads loaded", features=len(roads_gdf))

        logger.info("Loading target sites")
        sites = load_sites(cfg.inputs.target_sites, cfg.parameters.h3_resolution)
        logger.info("Sites loaded", count=len(sites))
        for site in sites:
            logger.debug("Site found", name=site.name, priority=site.priority)

    # Initialize elevation provider
    logger.info("[3/9] Loading elevation data")
    with PerfTimer("load_elevation"):
        elevation_provider = ElevationProvider(cfg.inputs.elevation)
    logger.info("Elevation provider initialized")

    # Load city boundary polygons (optional)
    city_polygons = []
    if cfg.inputs and cfg.inputs.city_boundaries:
        import geopandas as gpd
        logger.info("Loading city boundaries", path=cfg.inputs.city_boundaries)
        cb_gdf = gpd.read_file(cfg.inputs.city_boundaries)
        city_polygons = list(cb_gdf.geometry)
        logger.info("City boundaries loaded", count=len(city_polygons))

    # Generate H3 grid (only cells with roads)
    logger.info("[4/9] Generating H3 grid")
    with PerfTimer("generate_h3_grid"):
        cells = generate_road_grid(boundary, roads_gdf, elevation_provider,
                                   cfg.parameters, city_polygons=city_polygons)

    # Snap sites to nearest road cell
    logger.info("[4.5/9] Snapping sites to road cells")
    snap_sites_to_roads(sites, cells)

    # Create mesh surface
    logger.info("[5/9] Creating mesh surface")
    with PerfTimer("create_mesh_surface"):
        surface = MeshSurface(cells, cfg.parameters,
                              elevation_provider=elevation_provider)
    logger.info("Mesh surface created", cells=len(surface.cells))

    # Initialize LOS cache
    logger.info("[6/9] Initializing LOS cache")
    los_cache = LOSCache()
    logger.info("LOS cache initialized")

    # Build routing graph
    logger.info("[7/9] Building routing graph")
    with PerfTimer("build_routing_graph"):
        routing_graph = build_routing_graph(cells, roads_gdf, cfg.parameters)
    logger.info("Routing graph built",
                nodes=routing_graph.number_of_nodes(),
                edges=routing_graph.number_of_edges())

    # Connect sites by priority hierarchy
    logger.info("[8/10] Connecting sites by priority")
    with PerfTimer("connect_sites"):
        connect_sites_by_priority(sites, surface, routing_graph, los_cache)

    # Compute visibility edges between all towers
    logger.info("[9/10] Computing visibility edges")
    with PerfTimer("visibility_edges"):
        surface.update_visibility_edges(los_cache)
    logger.info("Visibility graph built",
                towers=surface.visibility_graph.tower_count(),
                edges=surface.visibility_graph.edge_count())

    # Compute per-cell coverage metrics
    logger.info("[9.5/10] Computing cell coverage")
    with PerfTimer("cell_coverage"):
        surface.compute_cell_coverage(los_cache)

    # Export results — use YAML output paths when configured, else --output dir
    logger.info("[10/10] Exporting results")
    with PerfTimer("export_results"):
        _defaults = OutputPaths()
        out = cfg.outputs

        towers_path = out.towers if out.towers != _defaults.towers else os.path.join(output, 'towers.geojson')
        coverage_path = out.coverage if out.coverage != _defaults.coverage else os.path.join(output, 'coverage.geojson')
        report_path = out.report if out.report != _defaults.report else os.path.join(output, 'report.json')
        edges_path = out.visibility_edges if out.visibility_edges != _defaults.visibility_edges else os.path.join(output, 'visibility_edges.geojson')

        # Ensure output directories exist
        for p in (towers_path, coverage_path, report_path, edges_path):
            os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)

        export_towers_geojson(surface, towers_path)
        export_coverage_geojson(surface, coverage_path)
        generate_report(surface, report_path)
        export_visibility_edges_geojson(surface, edges_path)
        if surface.gap_repair_debug:
            debug_path = os.path.join(output, 'gap_repair_hexes.geojson')
            export_gap_repair_hexes_geojson(surface.gap_repair_debug, debug_path)

    # Log cache stats
    cache_stats = los_cache.stats()
    logger.info("Cache statistics",
                los_entries=cache_stats['size'],
                los_hit_rate=f"{cache_stats['hit_rate']:.1%}")

    elev_stats = elevation_provider.cache_stats()
    logger.info("Elevation cache", **elev_stats)

    logger.info("Optimization complete",
                towers_placed=len(surface.towers),
                towers_path=towers_path,
                edges_path=edges_path)


if __name__ == '__main__':
    main()

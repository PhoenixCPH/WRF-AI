"""
Data preprocessing script for WRF files
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.config import Config
from data.preprocessing import WRFDataProcessor, setup_logging, get_memory_usage, get_gpu_memory_usage
from utils.utils import (
    validate_netcdf_file, create_directory_structure, 
    get_file_size, cleanup_memory
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess WRF data')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, default='data/raw',
                       help='Directory containing WRF data')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory for processed data')
    
    # Processing parameters
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Chunk size for processing')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for GPU processing')
    
    # File selection
    parser.add_argument('--file_pattern', type=str, default='wrfout_d01_*.nc',
                       help='File pattern to match')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to process')
    
    # GPU settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8,
                       help='GPU memory fraction to use')
    
    # Quality control
    parser.add_argument('--validate_files', action='store_true',
                       help='Validate NetCDF files before processing')
    parser.add_argument('--skip_invalid', action='store_true',
                       help='Skip invalid files')
    
    # Logging
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--log_file', type=str, default='logs/preprocessing.log',
                       help='Log file path')
    
    # Performance
    parser.add_argument('--monitor_memory', action='store_true',
                       help='Monitor memory usage')
    parser.add_argument('--cleanup_interval', type=int, default=50,
                       help='Memory cleanup interval')
    
    return parser.parse_args()


def process_single_file(args):
    """Process a single WRF file"""
    file_path, output_dir, config, device = args
    
    try:
        # Initialize processor for this worker
        processor = WRFDataProcessor(config)
        processor.device = torch.device(device)
        
        # Process file
        processed_data = processor.preprocess_single_file(file_path)
        
        # Save processed data
        filename = os.path.basename(file_path).replace('.nc', '_processed.pt')
        output_path = os.path.join(output_dir, filename)
        
        torch.save(processed_data.cpu(), output_path)
        
        return {
            'file_path': file_path,
            'status': 'success',
            'output_path': output_path,
            'file_size': get_file_size(file_path),
            'output_size': get_file_size(output_path)
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main preprocessing function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_file, args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting WRF data preprocessing")
    logger.info(f"Arguments: {args}")
    
    # Create configuration
    config = Config()
    config.data.data_dir = args.data_dir
    config.data.processed_dir = args.output_dir
    config.data.chunk_size = args.chunk_size
    config.data.batch_size = args.batch_size
    config.system.device = args.device
    config.system.gpu_memory_fraction = args.gpu_memory_fraction
    config.system.log_level = args.log_level
    config.system.log_file = args.log_file
    
    # Create directory structure
    create_directory_structure(args.output_dir)
    
    # Get list of files to process
    file_pattern = os.path.join(args.data_dir, args.file_pattern)
    file_list = glob.glob(file_pattern)
    file_list = sorted(file_list)
    
    if args.max_files:
        file_list = file_list[:args.max_files]
    
    logger.info(f"Found {len(file_list)} files to process")
    
    # Validate files if requested
    if args.validate_files:
        logger.info("Validating NetCDF files...")
        valid_files = []
        
        for file_path in file_list:
            if validate_netcdf_file(file_path):
                valid_files.append(file_path)
            else:
                logger.warning(f"Invalid file: {file_path}")
                if not args.skip_invalid:
                    raise ValueError(f"Invalid file found: {file_path}")
        
        file_list = valid_files
        logger.info(f"Found {len(file_list)} valid files")
    
    if not file_list:
        logger.error("No files to process")
        return
    
    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 8)
    
    logger.info(f"Using {args.num_workers} workers")
    
    # Process files
    processed_files = []
    failed_files = []
    
    try:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # Prepare arguments
            process_args = [
                (file_path, args.output_dir, config, args.device)
                for file_path in file_list
            ]
            
            # Submit tasks
            future_to_file = {
                executor.submit(process_single_file, arg): arg[0]
                for arg in process_args
            }
            
            # Process results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    if result['status'] == 'success':
                        processed_files.append(result)
                        logger.info(f"Processed: {file_path}")
                    else:
                        failed_files.append(result)
                        logger.error(f"Failed: {file_path} - {result['error']}")
                        
                except Exception as e:
                    failed_files.append({
                        'file_path': file_path,
                        'status': 'error',
                        'error': str(e)
                    })
                    logger.error(f"Failed: {file_path} - {e}")
                
                # Monitor memory
                if args.monitor_memory and len(processed_files) % args.cleanup_interval == 0:
                    memory = get_memory_usage()
                    gpu_memory = get_gpu_memory_usage()
                    
                    logger.info(f"Memory usage: CPU={memory['rss']:.2f}GB, GPU={gpu_memory}")
                    
                    # Cleanup
                    cleanup_memory()
    
    except KeyboardInterrupt:
        logger.info("Preprocessing interrupted by user")
    except Exception as e:
        logger.error(f"Preprocessing failed with error: {e}")
        raise
    
    # Summary
    logger.info(f"Preprocessing completed:")
    logger.info(f"  Successfully processed: {len(processed_files)}")
    logger.info(f"  Failed: {len(failed_files)}")
    
    if failed_files:
        logger.warning("Failed files:")
        for failed in failed_files:
            logger.warning(f"  {failed['file_path']}: {failed.get('error', 'Unknown error')}")
    
    # Save processing summary
    summary = {
        'total_files': len(file_list),
        'processed_files': len(processed_files),
        'failed_files': len(failed_files),
        'processing_details': processed_files + failed_files
    }
    
    summary_path = os.path.join(args.output_dir, 'preprocessing_summary.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
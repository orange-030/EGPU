# -*- coding: utf-8 -*-
# @Author  : Ran Wu
# @Time    : 2025//14 11:51

import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import logging
from datetime import datetime
import matplotlib.dates as mdates
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

# Configure logging for detailed runtime information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress non-critical pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


class EnvironmentalUncertaintyIndex:
    """Class to compute the Environmental Policy Uncertainty Index from text data."""

    def __init__(self, data_dir: str, file_list: List[str]):
        """
        Initialize the index computation with directory and source list.

        Args:
            data_dir (str): Path to the directory containing text files.
            file_list (List[str]): List of newspaper sources to process.
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.regex_patterns = self._compile_regex_patterns()

    @staticmethod
    def _compile_regex_patterns() -> dict:
        """
        Compile regex patterns for text matching to improve performance.

        Returns:
            dict: Dictionary of compiled regex patterns.
        """
        # Split environment terms into logical groups for clarity and validation
        environment_terms = [
            # Core environmental concepts
            '生态|雾霾|绿色|可持续|空气质量|污染|气候变化|节能|新能源|环境保护|排放|'
            '可再生能源|碳排放|环境影响|生态平衡|蓝天保卫战|生态环境|水质|空气污染',

            # Policy and initiative terms
            '柴油车治理|绿色低碳|清洁生产|黑臭水体|绿色产品|零排放|空气污染防治法|'
            '水功能|低碳发展|环境恶化|绿色发展|绿水青山|金山银山|推动绿色发展|'
            '可持续增长|美丽中国|污染减排|先污染后治理|能源革命|煤炭控制|碳达峰|'
            '绿色转型|淡水环境|京都议定书|巴黎协定|减排目标|环境卫星|环境公约|'
            '人与自然|绿色愿景|能源足迹|太阳能|太阳能发电|垃圾分类|强制环保|'
            '气候行动|环境健康危机|碳中和目标|碳中和承诺|绿色恢复|污染治理|水污染',

            # Conservation and biodiversity
            '三江源保护|国土绿化|全国绿色|温室气体|生态优先|循环发展|清山绿水|'
            '环境承载力|生命共同体|沙漠化|水土流失|湖泊|湿地|生物多样性|环境容量|'
            '生态安全|生态经济|可持续发展|生态城市|绿色供应链|绿色交通|绿色建筑|'
            '农业绿色发展|生态宜居|大气质量|土壤污染|净土保卫战|固体废物|海洋垃圾|'
            '零废城市|绿色矿山|保护修复|长江保护|黄河保护|渤海综合治理|海洋生态',

            # Disaster and resource management
            '防灾减灾|生态文化|富营养化|自然资源产权|自然资源使用|生态保护补偿|'
            '生态修复|五位一体|循环经济|两型社会|环境友好|限电|碧水保卫战|气候|'
            '自然灾害|粗放产业|自然生态禀赋|环境效益|绿色时尚|生态文明建设|国土空间',

            # Pollution and enforcement
            '工业污染|环境质量|企业污染治理|区域环境治理|污染物排放|生态破坏|'
            '生态系统功能|核与辐射|噪声|城市环境保护|减排|源头治理|低碳转型|'
            '环境监测|环保督察|环境研究|环境执法|环境教育|煤化减排|城市综合执法|'
            '固废|环境规划|联合国千年发展目标|人居环境|生态之美|保护与发展|储能',

            # Restoration and specific initiatives
            '清水长流|河道清理|荒山沙漠|环保检查|整改行动|森林抚育|自愿减排|'
            '一次性塑料|红树林|以竹代塑|栖息地|气候适应|沙漠化治理|进口废物|'
            '医疗废物|电子废物|铬渣|电离辐射|电磁辐射|环境投诉|外来物种|禁渔|'
            '绿色信贷|排污权|环境巡查|饮用水安全|清洁行动|城乡清洁|清洁工程|沼气|'
            '森林火灾|森林防火|环境标准|汞污染|二噁英|气象灾害|生态红线|自然保护区',

            # Additional terms
            '绿色屏障|秸秆回收|生态账本|零碳|绿化|森林城市|碳市场|地下水|高原生态|'
            '干旱|降雨|抗旱减灾|生态补水|碧海|非法采砂|林草资源|禁渔期|过剩产能'
        ]

        # Combine terms into a single regex with a non-capturing group
        environment_pattern = '|'.join(environment_terms)
        environment_regex = f'(?:{environment_pattern})'

        return {
            'uncertainty': re.compile(
                r'不确定|不明确|不稳定|未明|难以预料|难以预测|难以预计|难以估计'
            ),
            'policy': re.compile(
                r'能源部|生态环境部|自然资源部|国家发展改革委|水利部|农业农村部|国家能源局|'
                r'国家林业和草原局|法律|税收'
            ),
            'environment': re.compile(environment_regex)
        }

    def _process_file(self, txt_path: Path) -> Optional[dict]:
        """
        Process a single text file to extract date and content.

        Args:
            txt_path (Path): Path to the text file.

        Returns:
            Optional[dict]: Dictionary with date, text, and source, or None if invalid.
        """
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                logger.debug(f"Empty content in {txt_path}, skipping")
                return None

            # Extract date from filename (YYYY-MM-DD)
            date_str = txt_path.name[:10]
            try:
                date = pd.to_datetime(date_str, errors='raise')
            except ValueError:
                logger.warning(f"Invalid date format in {txt_path}: {date_str}, skipping")
                return None

            source = txt_path.parent.name
            return {'Date': date, 'Text': content, 'source': source}

        except Exception as e:
            logger.error(f"Error processing {txt_path}: {e}")
            return None

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess text data from specified sources.

        Returns:
            pd.DataFrame: DataFrame with Date, Text, and source columns.
        """
        logger.info("Starting data loading")
        results = []

        # Parallel file processing to improve performance
        with ThreadPoolExecutor() as executor:
            for source in self.file_list:
                root_folder = self.data_dir / source
                logger.info(f"Processing source: {source}")

                txt_files = list(root_folder.rglob('*.txt'))
                results.extend(
                    [result for result in executor.map(self._process_file, txt_files) if result is not None]
                )

                if len(results) % 1000 == 0:
                    logger.info(f"Processed {len(results)} files")

        df = pd.DataFrame(results)
        df = df.dropna(subset=['Date', 'Text']).reset_index(drop=True)

        logger.info(f"Total records: {len(df)}")
        if not df.empty:
            logger.info(f"Time range: {df['Date'].min().date()} to {df['Date'].max().date()}")

        return df

    def compute_index(
            self,
            df: pd.DataFrame,
            freq: str = 'D',
            eras: Optional[List[Tuple[datetime, datetime]]] = None,
            min_sample_size: int = 5
    ) -> pd.DataFrame:
        """
        Compute the Environmental Policy Uncertainty Index.

        Args:
            df (pd.DataFrame): Input DataFrame with Date, Text, and source columns.
            freq (str): Aggregation frequency ('D' for daily, 'M' for monthly).
            eras (Optional[List[Tuple[datetime, datetime]]]): List of (start, end) dates for normalization.
            min_sample_size (int): Minimum articles per period to compute score.

        Returns:
            pd.DataFrame: DataFrame with Date, combined, and combined_smoothed columns.
        """
        logger.info(f"Computing index with frequency: {freq}")

        # Calculate source weights based on article counts
        source_counts = df['source'].value_counts()
        source_weights = {src: 1 / (count / len(df)) for src, count in source_counts.items()}

        datas = []
        # Group by frequency and compute scores
        for date, period_df in df.groupby(pd.Grouper(key='Date', freq=freq)):
            if len(period_df) < min_sample_size:
                logger.debug(f"Skipping {date} due to insufficient samples ({len(period_df)} < {min_sample_size})")
                continue

            # Compute weighted matches
            matches = period_df.apply(
                lambda row: source_weights[row['source']] * (
                    1 if all(
                        pattern.search(row['Text'] or '')
                        for pattern in self.regex_patterns.values()
                    ) else 0
                ),
                axis=1
            )
            total_weight = sum(source_weights[src] for src in period_df['source'])
            raw_score = matches.sum() / total_weight if total_weight > 0 else 0
            datas.append({'Date': date, 'combined_raw': raw_score})

        result_df = pd.DataFrame(datas)

        if result_df.empty:
            logger.warning("No valid data after processing")
            return pd.DataFrame(columns=['Date', 'combined', 'combined_smoothed'])

        # Validate raw scores
        if (result_df['combined_raw'] < 0).any() or (result_df['combined_raw'] > 1).any():
            logger.error("Raw scores out of [0, 1] range")
            raise ValueError("Raw scores must be in [0, 1]")

        # Normalize scores by era or globally
        if eras:
            result_df['era'] = pd.NA
            for start, end in eras:
                mask = (result_df['Date'] >= start) & (result_df['Date'] <= end)
                result_df.loc[mask, 'era'] = f"{start.date()}_{end.date()}"

            normalized = []
            for era, era_df in result_df.groupby('era'):
                if era_df.empty:
                    continue
                min_score, max_score = era_df['combined_raw'].min(), era_df['combined_raw'].max()
                if max_score == min_score:
                    logger.warning(f"Constant raw scores in era {era}, setting normalized scores to 0")
                    era_df['combined'] = 0
                else:
                    era_df['combined'] = 100 * (era_df['combined_raw'] - min_score) / (max_score - min_score)
                normalized.append(era_df)

            result_df = pd.concat(normalized).sort_values('Date')
        else:
            min_score, max_score = result_df['combined_raw'].min(), result_df['combined_raw'].max()
            if max_score == min_score:
                logger.warning("Constant raw scores, setting normalized scores to 0")
                result_df['combined'] = 0
            else:
                result_df['combined'] = 100 * (result_df['combined_raw'] - min_score) / (max_score - min_score)

        # Ensure normalized scores are within [0, 100]
        result_df['combined'] = result_df['combined'].clip(lower=0, upper=100)

        # Apply smoothing
        window = 45 if freq == 'D' else 3
        result_df['combined_smoothed'] = result_df['combined'].rolling(
            window=window, min_periods=1
        ).mean()

        # Log statistics
        for era in result_df['era'].unique() if eras else ['all']:
            era_df = result_df[result_df['era'] == era] if eras else result_df
            if not era_df.empty:
                logger.info(
                    f"Era {era}: raw_mean={era_df['combined_raw'].mean():.4f}, "
                    f"raw_std={era_df['combined_raw'].std():.4f}, "
                    f"normalized_mean={era_df['combined'].mean():.2f}, "
                    f"normalized_std={era_df['combined'].std():.2f}"
                )

        return result_df[['Date', 'combined', 'combined_smoothed']]

    def plot_index(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot the computed index and save the figure.

        Args:
            df (pd.DataFrame): DataFrame with Date, combined, and combined_smoothed columns.
            output_dir (Path): Directory to save the plot.
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            df['Date'], df['combined'],
            label='Daily Environmental Uncertainty Index', alpha=0.5
        )
        ax.plot(
            df['Date'], df['combined_smoothed'],
            label='45-Day Smoothed Index', linewidth=2
        )

        ax.set_xlabel('Date')
        ax.set_ylabel('Index (0-100)')
        ax.set_title('Daily Environmental Policy Uncertainty Index')

        # Format date axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        fig.autofmt_xdate()

        ax.legend()
        ax.grid(True)

        output_path = output_dir / 'env_uncertainty_plot_daily.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        logger.info(f"Plot saved to {output_path}")


def main():
    """Main function to execute the index computation and visualization."""
    data_dir = '/Users/wuran/Desktop/报纸文本分析'
    file_list = ['光明日报', '经济日报', '南方周末', '人民日报', '中国青年报']

    # Initialize index calculator
    calculator = EnvironmentalUncertaintyIndex(data_dir, file_list)

    # Load data
    df = calculator.load_data()

    # Define eras for normalization
    eras = [
        (pd.to_datetime('2000-01-01'), pd.to_datetime('2015-12-31')),  # Kyoto to Paris
        (pd.to_datetime('2016-01-01'), pd.to_datetime('2020-09-30')),  # Paris to Carbon Neutrality
        (pd.to_datetime('2020-10-01'), pd.to_datetime('2023-12-31'))  # Post-Carbon Neutrality
    ]

    # Compute index
    combined_df = calculator.compute_index(df, freq='D', eras=eras, min_sample_size=5)

    # Save results
    output_path = Path(data_dir) / 'env_uncertainty_indices_daily.csv'
    combined_df.to_csv(output_path, index=False, date_format='%Y-%m-%d', encoding='utf-8-sig')
    logger.info(f"Results saved to {output_path}")

    # Plot results
    calculator.plot_index(combined_df, Path(data_dir))


if __name__ == '__main__':
    main()


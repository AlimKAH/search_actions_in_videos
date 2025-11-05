import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


class ResultAnalyzer:
    """Analyze video search results and generate insights."""

    def __init__(self, metadata_path: Path, output_dir: Path):
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df = None


    def load_data(self) -> pd.DataFrame:
        """Load metadata CSV and validate."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
        self.df = pd.read_csv(self.metadata_path)
        self.df['duration'] = self.df['end_time'] - self.df['start_time']
        return self.df


    def generate_summary_stats(self) -> Dict:
        """Generate summary statistics."""
        stats = {
            'total_fragments': len(self.df),
            'unique_queries': self.df['query'].nunique(),
            'unique_films': self.df['source_film'].nunique(),
            'avg_similarity': self.df['similarity_score'].mean(),
            'median_similarity': self.df['similarity_score'].median(),
            'std_similarity': self.df['similarity_score'].std(),
            'min_similarity': self.df['similarity_score'].min(),
            'max_similarity': self.df['similarity_score'].max(),
            'fragments_per_query': self.df.groupby('query').size().to_dict(),
            'fragments_per_film': self.df.groupby('source_film').size().to_dict(),
            'avg_score_per_query': self.df.groupby('query')['similarity_score'].mean().to_dict(),
        }
        return stats


    def plot_score_distribution(self):
        """Plot overall similarity score distribution."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['similarity_score'], bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(self.df['similarity_score'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["similarity_score"].mean():.3f}')
        plt.axvline(self.df['similarity_score'].median(), color='green', linestyle='--', label=f'Median: {self.df["similarity_score"].median():.3f}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Similarity Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot([self.df[self.df['query'] == q]['similarity_score'].values for q in self.df['query'].unique()],
                    labels=self.df['query'].unique())
        plt.ylabel('Similarity Score')
        plt.xlabel('Query')
        plt.title('Score Distribution by Query')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_query_performance(self):
        """Plot query-specific performance metrics."""
        query_stats = self.df.groupby('query').agg({
            'similarity_score': ['mean', 'std', 'count', 'min', 'max']
        }).round(3)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        queries = self.df['query'].unique()
        colors = sns.color_palette('husl', len(queries))
        
        ax = axes[0, 0]
        counts = self.df['query'].value_counts()
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel('Number of Fragments')
        ax.set_xlabel('Query')
        ax.set_title('Fragments Found per Query')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        means = self.df.groupby('query')['similarity_score'].mean().sort_values(ascending=False)
        ax.bar(means.index, means.values, color=colors)
        ax.set_ylabel('Average Similarity Score')
        ax.set_xlabel('Query')
        ax.set_title('Average Score by Query')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        for i, query in enumerate(queries):
            query_data = self.df[self.df['query'] == query]['similarity_score']
            ax.scatter([query] * len(query_data), query_data, alpha=0.6, s=30, color=colors[i], label=query)
        ax.set_ylabel('Similarity Score')
        ax.set_xlabel('Query')
        ax.set_title('Score Distribution (All Fragments)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        stds = self.df.groupby('query')['similarity_score'].std().sort_values(ascending=False)
        ax.bar(stds.index, stds.values, color=colors)
        ax.set_ylabel('Standard Deviation')
        ax.set_xlabel('Query')
        ax.set_title('Score Variability by Query')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'query_performance.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_temporal_distribution(self):
        """Plot temporal distribution of found fragments."""
        fig, axes = plt.subplots(len(self.df['query'].unique()), 1, figsize=(14, 4 * len(self.df['query'].unique())))
        
        if len(self.df['query'].unique()) == 1:
            axes = [axes]
        
        for idx, query in enumerate(self.df['query'].unique()):
            query_data = self.df[self.df['query'] == query]
            
            ax = axes[idx]
            for film in query_data['source_film'].unique():
                film_data = query_data[query_data['source_film'] == film]
                ax.scatter(film_data['start_time'] / 60, 
                          [film] * len(film_data),
                          c=film_data['similarity_score'],
                          cmap='RdYlGn',
                          s=100,
                          alpha=0.7,
                          vmin=self.df['similarity_score'].min(),
                          vmax=self.df['similarity_score'].max())
            
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Film')
            ax.set_title(f'Temporal Distribution: {query}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


    def plot_film_heatmap(self):
        """Plot heatmap of query performance across films."""
        pivot = self.df.pivot_table(
            values='similarity_score',
            index='source_film',
            columns='query',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, max(6, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.35, vmin=0.2, vmax=0.5)
        plt.title('Average Similarity Score: Films vs Queries')
        plt.xlabel('Query')
        plt.ylabel('Film')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'film_query_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()


    def detect_outliers(self) -> pd.DataFrame:
        """Detect potential false positives and true positives."""
        outliers = []
        
        for query in self.df['query'].unique():
            query_data = self.df[self.df['query'] == query]
            q1 = query_data['similarity_score'].quantile(0.25)
            q3 = query_data['similarity_score'].quantile(0.75)
            iqr = q3 - q1
            
            low_threshold = q1 - 1.5 * iqr
            high_threshold = q3 + 1.5 * iqr
            
            low_outliers = query_data[query_data['similarity_score'] < low_threshold]
            high_outliers = query_data[query_data['similarity_score'] > high_threshold]
            
            for _, row in low_outliers.iterrows():
                outliers.append({
                    'type': 'Low Score (Possible False Positive)',
                    'query': row['query'],
                    'film': row['source_film'],
                    'time': f"{row['start_time']:.0f}s",
                    'score': row['similarity_score']
                })
            
            for _, row in high_outliers.iterrows():
                outliers.append({
                    'type': 'High Score (Strong Match)',
                    'query': row['query'],
                    'film': row['source_film'],
                    'time': f"{row['start_time']:.0f}s",
                    'score': row['similarity_score']
                })
        
        return pd.DataFrame(outliers)


    def generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        avg_score = stats['avg_similarity']
        if avg_score < 0.30:
            recommendations.append(f"Average score is low ({avg_score:.3f}). Consider lowering similarity threshold.")
        elif avg_score > 0.45:
            recommendations.append(f"Average score is high ({avg_score:.3f}). Results are likely high quality.")
        
        for query, count in stats['fragments_per_query'].items():
            if count < 5:
                recommendations.append(f"Query '{query}' found only {count} fragments. May need threshold adjustment.")
            elif count > 50:
                recommendations.append(f"Query '{query}' found {count} fragments. Consider higher threshold to reduce false positives.")
        
        std = stats['std_similarity']
        if std > 0.15:
            recommendations.append(f"High score variance (σ={std:.3f}). Results quality varies significantly.")
        
        for query, avg in stats['avg_score_per_query'].items():
            if avg < stats['avg_similarity'] - 0.1:
                recommendations.append(f"Query '{query}' has below-average scores ({avg:.3f}). Check for false positives.")
        
        return recommendations


    def generate_report(self):
        """Generate comprehensive analysis report."""
        stats = self.generate_summary_stats()
        outliers = self.detect_outliers()
        recommendations = self.generate_recommendations(stats)
        
        report_lines = [
            "# Video Fragment Search Analysis Report",
            "",
            "## Summary Statistics",
            f"- Total Fragments: {stats['total_fragments']}",
            f"- Unique Queries: {stats['unique_queries']}",
            f"- Unique Films: {stats['unique_films']}",
            f"- Average Similarity: {stats['avg_similarity']:.3f}",
            f"- Median Similarity: {stats['median_similarity']:.3f}",
            f"- Std Dev: {stats['std_similarity']:.3f}",
            f"- Score Range: [{stats['min_similarity']:.3f}, {stats['max_similarity']:.3f}]",
            "",
            "## Fragments per Query",
        ]
        
        for query, count in sorted(stats['fragments_per_query'].items()):
            avg_score = stats['avg_score_per_query'][query]
            report_lines.append(f"- **{query}**: {count} fragments (avg score: {avg_score:.3f})")
        
        report_lines.extend([
            "",
            "## Top Performing Films",
        ])
        
        for film, count in sorted(stats['fragments_per_film'].items(), key=lambda x: x[1], reverse=True)[:5]:
            report_lines.append(f"- {film}: {count} fragments")
        
        if len(outliers) > 0:
            report_lines.extend([
                "",
                "## Outliers Detected",
                f"- Total Outliers: {len(outliers)}",
                f"- Low Scores: {len(outliers[outliers['type'].str.contains('Low')])}",
                f"- High Scores: {len(outliers[outliers['type'].str.contains('High')])}",
            ])
        
        report_lines.extend([
            "",
            "## Recommendations",
        ])
        
        for rec in recommendations:
            report_lines.append(f"- {rec}")
        
        report_lines.extend([
            "",
            "## Generated Plots",
            "1. `score_distribution.png` - Overall score distribution",
            "2. `query_performance.png` - Per-query performance metrics",
            "3. `temporal_distribution.png` - Timeline of found fragments",
            "4. `film_query_heatmap.png` - Cross-analysis heatmap",
            "",
        ])
        
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        if len(outliers) > 0:
            outliers.to_csv(self.output_dir / 'outliers.csv', index=False)
        
        return report_path


    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("Loading data...")
        self.load_data()
        
        print("Generating summary statistics...")
        stats = self.generate_summary_stats()
        
        print("Plotting score distribution...")
        self.plot_score_distribution()
        
        print("Plotting query performance...")
        self.plot_query_performance()
        
        print("Plotting temporal distribution...")
        self.plot_temporal_distribution()
        
        print("Plotting film-query heatmap...")
        self.plot_film_heatmap()
        
        print("Detecting outliers...")
        outliers = self.detect_outliers()
        
        print("Generating report...")
        report_path = self.generate_report()
        
        print(f"\nAnalysis complete!")
        print(f"  - Report: {report_path}")
        print(f"  - Plots: {self.output_dir}")
        print(f"  - Outliers: {len(outliers)} detected")
        
        return stats, outliers


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze video search results")
    parser.add_argument('--metadata', type=Path, default=Path('./output/metadata.csv'), help='Path to metadata CSV')
    parser.add_argument('--output', type=Path, default=Path('./analysis'), help='Output directory for analysis')
    
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(args.metadata, args.output)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()


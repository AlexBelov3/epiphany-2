# Set locale settings to suppress warning messages
Sys.setlocale("LC_CTYPE", "C")
Sys.setlocale("LC_COLLATE", "C")
Sys.setlocale("LC_TIME", "C")
Sys.setlocale("LC_MESSAGES", "C")
Sys.setlocale("LC_MONETARY", "C")
Sys.setlocale("LC_PAPER", "C")
Sys.setlocale("LC_MEASUREMENT", "C")

# Set CRAN mirror to avoid the CRAN mirror error
options(repos = c(CRAN = "https://cran.r-project.org"))
    if (!requireNamespace("TopDom", quietly = TRUE)) {
  install.packages("TopDom")
}
# Load necessary libraries
library(TopDom)

# Define the file paths
args <- commandArgs(trailingOnly = TRUE)
hic_matrix_path <- args[1]
output_data_path <- args[2]

# Load the Hi-C matrix
hic_matrix <- as.matrix(read.table(hic_matrix_path, header = FALSE, sep = "\t"))
print("Structure of 'hic_matrix':")
str(hic_matrix)
# Define chromosome and bin size
chr <- "chr1"
bin_size <- 10000  # Adjust based on your actual bin size
n_bins <- nrow(hic_matrix)

# Create the bins data frame
bins <- data.frame(
  id = 1:n_bins,
  chr = rep(chr, n_bins),
  from.coord = seq(0, by = bin_size, length.out = n_bins),
  to.coord = seq(bin_size, by = bin_size, length.out = n_bins)
)

# Create the TopDomData object
topdom_data <- list(bins = bins, counts = hic_matrix)
class(topdom_data) <- "TopDomData"

# Check the structure of topdom_data
print("Structure of 'topdom_data':")
str(topdom_data)

# Save the bins and counts to separate files for Python processing
write.table(bins, file = paste0(output_data_path, "_bins.tsv"), sep = "\t", row.names = FALSE, col.names = TRUE)
write.table(hic_matrix, file = paste0(output_data_path, "_counts.tsv"), sep = "\t", row.names = FALSE, col.names = FALSE)

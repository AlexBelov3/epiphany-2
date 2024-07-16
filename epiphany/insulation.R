# Load necessary libraries
library(TopDom)

# Define the file paths
args <- commandArgs(trailingOnly = TRUE)
hic_matrix_path <- args[1]
output_data_path <- args[2]

# Load the Hi-C matrix
hic_matrix <- as.matrix(read.table(hic_matrix_path, header = FALSE, sep = "\t"))

# Define chromosome and bin size
chr <- "chr1"
bin_size <- 40000  # Adjust based on your actual bin size
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